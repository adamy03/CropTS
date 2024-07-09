import ee
import geemap
import sys

from .data_utils import *


# =============S1-SAR================
def toNatural(img):
    return (
        ee.Image(10.0)
        .pow(img.select("..").divide(10.0))
        .copyProperties(img, ["system:time_start"])
    )


def toDB(img):
    return (
        ee.Image(img).log10().multiply(10.0).copyProperties(img, ["system:time_start"])
    )


def maskEdge(img):
    mask = (
        img.select(0)
        .unitScale(-25, 5)
        .multiply(255)
        .toByte()
        .connectedComponents(ee.Kernel.rectangle(1, 1), 100)
    )
    return img.updateMask(mask.select(0).abs())


def get_range_average(date_range, fc, bands=["VV", "VH"]):
    # Olha Danylo's procedure to create weekly means (adapted)
    dstamp = ee.Date(ee.List(date_range).get(0)).format("YYYYMMdd")
    date_range_start = ee.List(date_range).get(0)
    date_range_end = ee.List(date_range).get(1)
    col_title = [ee.String(band + "_").cat(dstamp) for band in bands]

    temp_collection = (
        fc.filterDate(date_range_start, date_range_end).mean().select(bands, col_title)
    )

    return temp_collection


def generate_s1_averages(roi, start_date, end_date, step):
    s1 = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterMetadata("instrumentMode", "equals", "IW")
        .filter(ee.Filter.eq("transmitterReceiverPolarisation", ["VV", "VH"]))
        .filterBounds(roi)
        .filterDate(start_date, end_date)
        .sort("system:time")
    )
    
    s1 = s1.map(maskEdge)
    s1 = s1.map(toNatural)

    days = ee.List.sequence(
        0, ee.Date(end_date).difference(ee.Date(start_date), "day"), step
    ).map(lambda d: ee.Date(start_date).advance(d, "day"))
    dates = days.slice(0, -1).zip(days.slice(1))
    s1res = dates.map(
        lambda range: s1.filterDate(ee.List(range).get(0), ee.List(range).get(1))
        .mean()
        .select(
            ["VV", "VH"],
            [
                ee.String("VV_").cat(ee.Date(ee.List(range).get(0)).format("YYYYMMdd")),
                ee.String("VH_").cat(ee.Date(ee.List(range).get(0)).format("YYYYMMdd")),
            ],
        )
    )
    s1res = s1res.map(toDB)
    s1stack = ee.ImageCollection(s1res).toBands()

    return s1stack


def extract_histar(roi, start_date, end_date, step):
    # Load the Landsat image collection
    GF_Landsat = (
        ee.ImageCollection("projects/ee-kalman-gap-filled/assets/histarfm_v5")
        .filterBounds(roi)
        .filterDate(start_date, end_date)
    )

    months = GF_Landsat.aggregate_array("month").distinct().sort()

    def EuropeanMosaic(num):
        ic = GF_Landsat.filter(ee.Filter.eq("month", num))
        img = ic.mosaic().selfMask()
        return img.copyProperties(ic.first(), ["system:time_start", "month", "year"])

    # Create European mosaic for each month
    GF_Landsat = ee.ImageCollection(months.map(EuropeanMosaic))

    # Sort the ImageCollection by time
    GF_Landsat = GF_Landsat.sort("system:time_start")

    # Function to scale and mask the error bands
    def scaleError(img):
        y = ee.Number.parse(img.get("year"))
        m = ee.Number.parse(img.get("month"))
        d = ee.Date.fromYMD(y, m, 15)
        doy = d.getRelative("day", "year").add(1)
        scaled = img.select(["P.*"]).multiply(0.5)
        return (
            img.addBands(scaled, None, True)
            .set({"month": m, "year": y, "DOY": doy})
            .copyProperties(img, ["system:time_start"])
        )

    GF_Landsat = GF_Landsat.map(scaleError)

    # Create a dummy collection with regular time space
    def RegTimeColl(date):
        DOY = ee.Date(date).getRelative("day", "year").add(1)
        m = ee.Number(ee.Date(date).get("month"))
        return (
            ee.Image(DOY)
            .rename("DOY")
            .int16()
            .set({"dummy": True, "system:time_start": date, "DOY": DOY, "month": m})
        )

    # Define the desired temporal resolution
    tUnit = "days"
    tResolution = step

    # Create a list of DOYs
    dateini = ee.Date(start_date)
    dateend = ee.Date(end_date)
    nSteps = dateend.difference(dateini, tUnit).divide(tResolution).floor()
    steps = ee.List.sequence(0, nSteps)
    timeRange = steps.map(
        lambda i: dateini.advance(ee.Number(i).multiply(tResolution), tUnit).millis()
    )

    # Create a dummy collection with regular time space
    tenDaysRes = ee.ImageCollection(timeRange.map(RegTimeColl))

    # Join the dummy collection with the Landsat collection
    join = ee.Join.saveAll("LS", "system:time_start", True, None, True)
    nDays = 31
    maxMilliDif = nDays * 24 * 3600 * 1000
    filter = ee.Filter.maxDifference(
        maxMilliDif, "system:time_start", None, "system:time_start"
    )
    tenDaysRes = ee.ImageCollection(join.apply(tenDaysRes, GF_Landsat, filter))

    # Define the interpolation function
    init_doy = ee.Image(GF_Landsat.first()).get("DOY")
    end_doy = ee.Image(GF_Landsat.sort("system:time_start", False).first()).get("DOY")

    def CompositeInterpolate(img):
        list = ee.List(img.get("LS"))
        LSbefore = ee.Image(
            ee.Algorithms.If(list.length().lt(2), list.get(0), list.get(-2))
        )
        LSafter = ee.Image(
            ee.Algorithms.If(list.length().lt(2), list.get(0), list.get(-1))
        )
        day = ee.Number(img.get("DOY"))
        ini = ee.Number(LSbefore.get("DOY"))
        end = ee.Number(LSafter.get("DOY"))
        tdiff = end.subtract(ini)
        slope = LSafter.subtract(LSbefore).divide(tdiff)
        img_inter = slope.multiply(day.subtract(ini)).add(LSbefore)
        img_inter = img_inter.addBands(img_inter.select("P.*").int16(), None, True)
        img_inter = img_inter.where(
            day.lte(init_doy),
            GF_Landsat.filter(ee.Filter.eq("DOY", ee.Number(init_doy))).first(),
        )
        img_inter = img_inter.where(
            day.gte(end_doy),
            GF_Landsat.filter(ee.Filter.eq("DOY", ee.Number(end_doy))).first(),
        )
        time = img.get("system:time_start")
        month = img.get("month")
        year = img.get("year")

        return img_inter.set(
            {
                "day": day,
                "month": month,
                "date_ini": ini,
                "date_end": end,
                "system:time_start": time,
                "year": year,
            }
        )

    # Apply the interpolation function to the dummy collection
    GF_Landsat = tenDaysRes.map(CompositeInterpolate)

    # Define the start and end dates
    start_date = start_date
    end_date = end_date

    # Extract the vis bands from HISTARFM
    histar = (
        GF_Landsat.filterDate(start_date, end_date)
        .filterBounds(roi)
        .sort("system:time")
    )
    days = ee.List.sequence(
        0, ee.Date(end_date).difference(ee.Date(start_date), "day"), step
    ).map(lambda d: ee.Date(start_date).advance(d, "day"))
    dates = days.slice(0, -1).zip(days.slice(1))

    histar = dates.map(
        lambda range: histar.filterDate(ee.List(range).get(0), ee.List(range).get(1))
        .mean()
        .select(
            [
                "B1_mean_post",
                "B2_mean_post",
                "B3_mean_post",
                "B4_mean_post",
                "B5_mean_post",
                "B7_mean_post",
            ],
            [
                ee.String("B1_mean_post_").cat(
                    ee.Date(ee.List(range).get(0)).format("YYYYMMdd")
                ),
                ee.String("B2_mean_post_").cat(
                    ee.Date(ee.List(range).get(0)).format("YYYYMMdd")
                ),
                ee.String("B3_mean_post_").cat(
                    ee.Date(ee.List(range).get(0)).format("YYYYMMdd")
                ),
                ee.String("B4_mean_post_").cat(
                    ee.Date(ee.List(range).get(0)).format("YYYYMMdd")
                ),
                ee.String("B5_mean_post_").cat(
                    ee.Date(ee.List(range).get(0)).format("YYYYMMdd")
                ),
                ee.String("B7_mean_post_").cat(
                    ee.Date(ee.List(range).get(0)).format("YYYYMMdd")
                ),
            ],
        )
    )

    histar = ee.ImageCollection(histar).toBands()

    return histar


def generate_fused(
    roi,
    start_date,
    end_date,
    step=10,
    export_scale=10,
):
    """Creates fused data, in order of bands 'B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'VH', and 'VV'

    Args:
        roi (_type_): region to extract from 
        start_date (_type_): _description_
        end_date (_type_): _description_
        step (int, optional): _description_. Defaults to 10.
        export_scale (int, optional): _description_. Defaults to 10.

    Returns:
        _type_: _description_
    """
    lucas_subset =  (
        ee.FeatureCollection('projects/ee-ayang115t/assets/lucas_2018_filtered_polygons')
        .filterBounds(roi)
    )
    s1 = generate_s1_averages(roi, start_date, end_date, step)
    histar = extract_histar(roi, start_date, end_date, step)

    fused = s1.addBands(histar)
    
    to_export = fused.sampleRegions(
        collection = lucas_subset,
        properties = ['POINT_ID','stratum'],
        tileScale = 16,
        scale = export_scale,
        geometries=False
    )

    fused_df = ee.data.computeFeatures({
        'expression': to_export,
        'fileFormat': 'PANDAS_DATAFRAME'
    })
    
    labels = pd.read_csv('/scratch/bbug/ayang1/raw_data/lucas/lucas_2018/copernicus_filtered/lucas_2018_filtered.csv')
    fused_df = add_lucas_labels(fused_df, labels)
    fused_df.drop(['geo', 'POINT_ID'], axis=1, inplace=True)
    fused_df = fused_df.loc[fused_df['LABEL']!='NOT_CROP']
    
    bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'VH', 'VV']
    band_df = []
    labels = fused_df['LABEL'].to_numpy()

    for band in bands:
        df = fused_df.loc[:, [i for i in filter(lambda x: band in x, fused_df.columns)]]
        col_names = list(df.columns)
        col_names.sort(key=lambda x: int(x.split('_')[0]))
        df = df.reindex(col_names, axis=1)
        band_df.append(np.expand_dims(df.to_numpy(), 1))
        
    data = np.concatenate(band_df, axis=1)
    
    return data, labels, to_export, fused_df
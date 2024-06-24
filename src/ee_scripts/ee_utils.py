import ee
import geemap

# =============S1-SAR================
def toNatural(img):
    return ee.Image(10.0).pow(img.select('..').divide(10.0)).copyProperties(img, ['system:time_start'])

def toDB(img):
    return ee.Image(img).log10().multiply(10.0).copyProperties(img, ['system:time_start'])

def maskEdge(img):
    mask = img.select(0).unitScale(-25, 5).multiply(255).toByte().connectedComponents(ee.Kernel.rectangle(1, 1), 100)
    return img.updateMask(mask.select(0).abs())

def get_range_average(date_range, fc, bands=['VV', 'VH']):
    # Olha Danylo's procedure to create weekly means (adapted)
    dstamp = ee.Date(ee.List(date_range).get(0)).format('YYYYMMdd')
    date_range_start = ee.List(date_range).get(0)
    date_range_end = ee.List(date_range).get(1)
    col_title = [ee.String(band+'_').cat(dstamp) for band in bands]

    temp_collection = fc.filterDate(date_range_start, date_range_end).mean().select(
        bands, 
        col_title
        )
    
    return temp_collection

def generate_s1_averages(
    start_date,
    end_date,
    aoi,
    step
):
    # Filter and preprocess the SAR images
    s1 = ee.ImageCollection('COPERNICUS/S1_GRD').filterMetadata('instrumentMode', 'equals', 'IW') \
        .filterDate(start_date, end_date) \
        .filterBounds(aoi) \
        .filter(ee.Filter.eq('transmitterReceiverPolarisation', ['VV', 'VH'])) \
        .sort('system:time')

    s1 = s1.map(maskEdge)
    s1 = s1.map(toNatural)

    # Define the date ranges
    days = ee.List.sequence(0, ee.Date(end_date).difference(ee.Date(start_date), 'day'), step) \
        .map(lambda d: ee.Date(start_date).advance(d, "day"))
    dates = days.slice(0, -1).zip(days.slice(1))

    # Extract the SAR dB for each date range

    s1 = dates.map(lambda x: get_range_average(x, s1, ['VV_', 'VH_']))
    s1 = ee.ImageCollection(s1.map(toDB)).toBands() 
    
    return s1

def sample_sar(feature, s1, scale, pixel_limit, random_seed):
    code = feature.get('Afgkode')
    desc = feature.get('Afgroede')
    sampled = s1.sample(
        region=feature.geometry(),
        scale=scale,
        numPixels=pixel_limit,
        seed=random_seed,
        geometries=False
    )

    sampled = sampled.map(lambda x: x.set('code', code))
    sampled = sampled.map(lambda x: x.set('crop', desc))

    return sampled
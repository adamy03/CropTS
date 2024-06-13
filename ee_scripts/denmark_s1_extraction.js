
// Env vars
var start_date='2018-01-01';
var end_date='2019-01-01'; 
var denmark_bounds = countries.filter(ee.Filter.eq('ADM0_NAME', 'Denmark'))
var step = 10 //date range for collected averages
var pixel_limit = 5 //number of pixels to sample per feature
var scale = 10 //resolution (meters) to sample in
var dataset_properties = ['Afgkode', 'Afgroede'] //dataset labels to copy to SAR samples-

// Denmark dataset
var aoi = denmark_pts.filterBounds(geometry)

// Extract SAR dB 
function toNatural(img) {
  return ee.Image(10.0).pow(img.select('..').divide(10.0)).copyProperties(img, ['system:time_start'])
}

function toDB(img) {
  return ee.Image(img).log10().multiply(10.0).copyProperties(img, ['system:time_start']);
}

function maskEdge(img) {
  var mask = img.select(0).unitScale(-25, 5).multiply(255).toByte().connectedComponents(ee.Kernel.rectangle(1,1), 100);
 return img.updateMask(mask.select(0).abs());  
}

function S1VHVV (img) { var ratio = img.select(['VH']).divide(img.select(['VV'])).rename('VHVV');
  return img.addBands(ratio,['VHVV'])
} 

var s1 = ee.ImageCollection('COPERNICUS/S1_GRD').filterMetadata('instrumentMode', 'equals', 'IW').
  filter(ee.Filter.eq('transmitterReceiverPolarisation', ['VV', 'VH'])).
  filterBounds(denmark_bounds).
  filterDate(start_date, end_date)
  .sort('system:time');

s1 = s1.map(maskEdge)
s1 = s1.map(toNatural)

var days = ee.List.sequence(0, ee.Date(end_date).difference(ee.Date(start_date), 'day'), step).
  map(function(d) { return ee.Date(start_date).advance(d, "day") })
var dates = days.slice(0,-1).zip(days.slice(1))

var s1 = dates.map(function(range) {
  var dstamp = ee.Date(ee.List(range).get(0)).format('YYYYMMdd')
  var temp_collection = s1.filterDate(ee.List(range).get(0),
  ee.List(range).get(1)).mean().select(['VV', 'VH'], [ee.String('VV_').cat(dstamp), ee.String('VH_').cat(dstamp)])
  return temp_collection
})

s1=s1.map(toDB)

var s1 = ee.ImageCollection(s1).toBands();

// Per feature extraction
function sampleSAR(feature) {
  var sampled_sar = s1.sample({
    region:feature.geometry(),
    scale:scale,
    numPixels:pixel_limit,
    seed:1,
    geometries:true
  })
  var sampled_sar = sampled_sar.map(function(sar_feature){
    return sar_feature.copyProperties(feature, dataset_properties)
  })
  
  return sampled_sar;
} 

var samples = aoi.map(sampleSAR)
var samples = samples.flatten()

print('samples', samples)

// Mapping samples
print('s1_10day_avg', s1)
print('aoi', aoi)
Map.centerObject(geometry);
Map.addLayer(aoi, {color:'blue'}, 'aoi')
Map.addLayer(samples.geometry(), {color:'red'}, 'samples')
Map.addLayer(s1.clip(aoi), {bands:['0_VV_20180101'], min:-25, max:-1, palette:['black', 'white']}, 'S1')

var countries = ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level0")
var denmark_pts = ee.FeatureCollection("projects/ee-ayang115t/assets/Marker_2018")
var geometry = ee.Geometry.Polygon() // must define before running script

// ================Env vars================
var start_date='2018-01-01';
var end_date='2019-01-01'; 
var denmark_bounds = countries.filter(ee.Filter.eq('ADM0_NAME', 'Denmark'))
var step = 10 //date range for collected averages
var pixel_limit = 1 //number of pixels to sample per feature
var scale = 100 //resolution (meters) to sample in
var dataset_properties = ['Afgkode', 'Afgroede'] //dataset labels to copy to SAR samples-

// ================Denmark dataset================
var aoi = denmark_pts.filterBounds(geometry)

// ================Extract SAR dB================
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
  filterDate(start_date, end_date).
  filterBounds(denmark_bounds).
  filter(ee.Filter.eq('transmitterReceiverPolarisation', ['VV', 'VH'])).
  sort('system:time');

s1 = s1.map(maskEdge)
s1 = s1.map(toNatural)

var days = ee.List.sequence(0, ee.Date(end_date).difference(ee.Date(start_date), 'day'), step).
  map(function(d){
    return ee.Date(start_date).advance(d, "day") 
  });
var dates = days.slice(0,-1).zip(days.slice(1));

var s1 = dates.map(function(range) {
  var dstamp = ee.Date(ee.List(range).get(0)).format('YYYYMMdd')
  var temp_collection = s1.filterDate(ee.List(range).get(0),
  ee.List(range).get(1)).mean().select(['VV', 'VH'], [ee.String('VV_').cat(dstamp), ee.String('VH_').cat(dstamp)])
  return temp_collection
});

s1 = ee.ImageCollection(s1.map(toDB)).toBands();

// ================Per feature extraction================
function sampleSAR(feature) {
  var code = feature.get('Afgkode');
  var desc = feature.get('Afgroede')
  var sampled = s1.sample({
    region:feature.geometry(),
    scale:scale,
    numPixels:pixel_limit,
    seed:1,
    geometries:false
  });
  
  sampled = sampled.map(function(feature){
    feature = feature.set('code', code);
    feature = feature.set('crop', desc)
    return feature
  })
  
  return sampled;
} 

var sampled_sar = ee.FeatureCollection(aoi.map(sampleSAR));
var sampled_sar = sampled_sar.flatten()

// ================Mapping samples================
// print('s1_10day_avg', s1)
print('aoi size', aoi.size())
print('sampled_sar size', sampled_sar.size())
print('sampled_sar', sampled_sar.limit(10))

Map.centerObject(geometry);
// Map.addLayer(aoi, {color:'blue'}, 'aoi')
// Map.addLayer(sampled_sar.geometry(), {color:'red'}, 'samples')
// Map.addLayer(s1.clip(aoi), {bands:['0_VV_20180101'], min:-25, max:-1, palette:['black', 'white']}, 'S1')

// ================Exporting samples================
Export.table.toDrive({
  collection: sampled_sar,
  description: 'S1_point_'+step+'days_'+scale+'m_2018_denmark_'+pixel_limit+'per',
  fileNamePrefix: 'S1_point_'+step+'days_'+scale+'m_2018_denmark_'+pixel_limit+'per',
  fileFormat: 'CSV',
  folder: 'GEE'
});

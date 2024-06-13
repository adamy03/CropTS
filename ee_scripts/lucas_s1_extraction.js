var lucas = ee.FeatureCollection('projects/ee-ayang115t/assets/lucas_2018_filtered_polygons')
var countries = ee.FeatureCollection('FAO/GAUL_SIMPLIFIED_500m/2015/level0');
// ============== Parameters ==============
// Date
var start_date='2018-01-01';
var end_date='2018-12-31';  

// Time step
var step = 10;

// Pixel spacing
var pix_export=10;

// Country subset
var country = 'Romania'
var country_bounds = countries.filter(ee.Filter.eq('ADM0_NAME', country))
Map.addLayer(country_bounds.draw('blue'), {opacity: 0.4}, 'Country Boundary')

// ============== Filter LUCAS ==============

var lucas_subset = lucas_points.filterBounds(country_bounds);

// ============== Extract S1 SAR ==============
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

function stack(i1, i2)
{
  return ee.Image(i1).addBands(ee.Image(i2))
}

var s1 = ee.ImageCollection('COPERNICUS/S1_GRD').filterMetadata('instrumentMode', 'equals', 'IW').
  filter(ee.Filter.eq('transmitterReceiverPolarisation', ['VV', 'VH'])).
  filterBounds(lucas_subset).filterDate(start_date, end_date)
  .sort('system:time');

print('Size', s1.size())

s1 = s1.map(maskEdge)
s1 = s1.map(toNatural)

print(s1.limit(10))

var days = ee.List.sequence(0, ee.Date(end_date).difference(ee.Date(start_date), 'day'), step).
  map(function(d) { return ee.Date(start_date).advance(d, "day") })

var dates = days.slice(0,-1).zip(days.slice(1))

var s1res = dates.map(function(range) {
  var dstamp = ee.Date(ee.List(range).get(0)).format('YYYYMMdd')
  var temp_collection = s1.filterDate(ee.List(range).get(0),
  ee.List(range).get(1)).mean().select(['VV', 'VH'], [ee.String('VV_').cat(dstamp), ee.String('VH_').cat(dstamp)])
  return temp_collection
})

s1res=s1res.map(toDB)
s1 = s1.map(S1VHVV)

var s1resRatio = dates.map(function(range) {
  var dstamp = ee.Date(ee.List(range).get(0)).format('YYYYMMdd')
  var temp_collection = s1.filterDate(ee.List(range).get(0),
  ee.List(range).get(1)).mean().select(['VHVV'], [ee.String('VHVV_').cat(dstamp)])
  return temp_collection
})

//put the two lists together 
var combine_db_ratio = s1res.zip(s1resRatio).flatten()

// Convert ImageCollection to image stack
var s1stack = combine_db_ratio.slice(1).iterate(stack, combine_db_ratio.get(0))

//transform the image to float to reduce size
s1stack = ee.Image(s1stack).toFloat()
Map.addLayer(s1stack.clip(country_bounds),{},'s1stack subset')
// ============== Export ==============

function export_map(stack, region, name) {
  var to_export = stack.sampleRegions({
  collection: region,
  properties: ['POINT_ID','stratum'],
  tileScale:16,
  scale: pix_export,
  geometries:false
  });
  
  Export.table.toDrive({'collection': to_export, 
  'description': 'S1_point_'+step+'days_'+pix_export+'m_1Jan-31Dec_'+name, 
  'fileNamePrefix':'S1_point_'+step+'days_'+pix_export+'m_1Jan-31Dec_'+name+'_ratio-db',
  folder:'GEE'
  })
}

export_map(s1stack, lucas_subset, country)


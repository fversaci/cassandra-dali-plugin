//////////////////////////////////////////////////////////////////////
// Pascal2 dataset: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit

CREATE KEYSPACE pascal2 WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};

// Tables with original files

// Mapping patch uuid to data and label
CREATE TABLE pascal2.data_orig(
  patch_id uuid,
  label blob,
  data blob,
  PRIMARY KEY ((patch_id))
);

// Mapping patch uuid to metadata
CREATE TABLE pascal2.metadata_orig(
  filename text,
  patch_id uuid,
  PRIMARY KEY ((patch_id))
);

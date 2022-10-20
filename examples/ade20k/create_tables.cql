//////////////////////////////////////////////////////////////////////
// ADE20K Outdoors dataset: https://www.kaggle.com/datasets/residentmario/ade20k-outdoors

CREATE KEYSPACE ade20k WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};

// Tables with original files

// Mapping patch uuid to data and label
CREATE TABLE ade20k.data_orig(
  patch_id uuid,
  label blob,
  data blob,
  PRIMARY KEY ((patch_id))
);

// Mapping patch uuid to metadata
CREATE TABLE ade20k.metadata_orig(
  filename text,
  patch_id uuid,
  PRIMARY KEY ((patch_id))
);

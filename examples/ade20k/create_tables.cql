//////////////////////////////////////////////////////////////////////
// ADE20K Outdoors dataset: https://www.kaggle.com/datasets/residentmario/ade20k-outdoors

CREATE KEYSPACE IF NOT EXISTS ade20k WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};

// Tables with original files

// Mapping patch uuid to data and label
CREATE TABLE IF NOT EXISTS ade20k.data(
  id uuid,
  label blob,
  data blob,
  PRIMARY KEY ((id))
);

// Mapping patch uuid to metadata
CREATE TABLE IF NOT EXISTS ade20k.metadata(
  filename text,
  id uuid,
  PRIMARY KEY ((id))
);

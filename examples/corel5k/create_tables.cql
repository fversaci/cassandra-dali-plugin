//////////////////////////////////////////////////////////////////////
// Corel-5k dataset: https://www.kaggle.com/datasets/parhamsalar/corel5k

CREATE KEYSPACE IF NOT EXISTS corel5k WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};

// Tables with original files

// Mapping patch uuid to data and label
CREATE TABLE IF NOT EXISTS corel5k.data(
  patch_id uuid,
  label blob,
  data blob,
  PRIMARY KEY ((patch_id))
);

// Mapping patch uuid to metadata
CREATE TABLE IF NOT EXISTS corel5k.metadata(
  filename text,
  patch_id uuid,
  PRIMARY KEY ((patch_id))
);

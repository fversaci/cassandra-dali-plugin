//////////////////////////////////////////////////////////////////////
// imagenet: https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data

CREATE KEYSPACE IF NOT EXISTS imagenet WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};

// Tables with original files

// Mapping patch uuid to data and label
CREATE TABLE IF NOT EXISTS imagenet.data_train(
  patch_id uuid,
  label int,
  data blob,
  PRIMARY KEY ((patch_id))
) WITH compression = {'enabled':'false'};

// Mapping patch uuid to metadata
CREATE TABLE IF NOT EXISTS imagenet.metadata_train(
  label int,
  or_label text,
  or_split text,
  patch_id uuid,
  PRIMARY KEY ((patch_id))
);

// Mapping patch uuid to data and label
CREATE TABLE IF NOT EXISTS imagenet.data_val(
  patch_id uuid,
  label int,
  data blob,
  PRIMARY KEY ((patch_id))
) WITH compression = {'enabled':'false'};

// Mapping patch uuid to metadata
CREATE TABLE IF NOT EXISTS imagenet.metadata_val(
  label int,
  or_label text,
  or_split text,
  patch_id uuid,
  PRIMARY KEY ((patch_id))
);


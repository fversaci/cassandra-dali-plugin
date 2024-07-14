//////////////////////////////////////////////////////////////////////
// imagenet: https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data

CREATE KEYSPACE IF NOT EXISTS imagenet WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};

// Tables with original files

// Mapping patch uuid to data and label
CREATE TABLE IF NOT EXISTS imagenet.data_train_orig(
  patch_id uuid,
  label int,
  data blob,
  PRIMARY KEY ((patch_id))
) WITH compression = {'enabled':'false'};

// Mapping patch uuid to metadata
CREATE TABLE IF NOT EXISTS imagenet.metadata_train_orig(
  label int,
  or_label text,
  or_split text,
  patch_id uuid,
  PRIMARY KEY ((patch_id))
);

// Mapping patch uuid to data and label
CREATE TABLE IF NOT EXISTS imagenet.data_val_orig(
  patch_id uuid,
  label int,
  data blob,
  PRIMARY KEY ((patch_id))
) WITH compression = {'enabled':'false'};

// Mapping patch uuid to metadata
CREATE TABLE IF NOT EXISTS imagenet.metadata_val_orig(
  label int,
  or_label text,
  or_split text,
  patch_id uuid,
  PRIMARY KEY ((patch_id))
);

// Tables with resized JPEGs


// Mapping patch uuid to data and label
CREATE TABLE IF NOT EXISTS imagenet.data_train_512_jpg(
  patch_id uuid,
  label int,
  data blob,
  PRIMARY KEY ((patch_id))
) WITH compression = {'enabled':'false'};

// Mapping patch uuid to metadata
CREATE TABLE IF NOT EXISTS imagenet.metadata_train_512_jpg(
  label int,
  or_label text,
  or_split text,
  patch_id uuid,
  PRIMARY KEY ((patch_id))
);

// Mapping patch uuid to data and label
CREATE TABLE IF NOT EXISTS imagenet.data_val_512_jpg(
  patch_id uuid,
  label int,
  data blob,
  PRIMARY KEY ((patch_id))
) WITH compression = {'enabled':'false'};

// Mapping patch uuid to metadata
CREATE TABLE IF NOT EXISTS imagenet.metadata_val_512_jpg(
  label int,
  or_label text,
  or_split text,
  patch_id uuid,
  PRIMARY KEY ((patch_id))
);

// Mapping patch uuid to data and label
CREATE TABLE IF NOT EXISTS imagenet.data_train_256_jpg(
  patch_id uuid,
  label int,
  data blob,
  PRIMARY KEY ((patch_id))
) WITH compression = {'enabled':'false'};

// Mapping patch uuid to metadata
CREATE TABLE IF NOT EXISTS imagenet.metadata_train_256_jpg(
  label int,
  or_label text,
  or_split text,
  patch_id uuid,
  PRIMARY KEY ((patch_id))
);

// Mapping patch uuid to data and label
CREATE TABLE IF NOT EXISTS imagenet.data_val_256_jpg(
  patch_id uuid,
  label int,
  data blob,
  PRIMARY KEY ((patch_id))
) WITH compression = {'enabled':'false'};

// Mapping patch uuid to metadata
CREATE TABLE IF NOT EXISTS imagenet.metadata_val_256_jpg(
  label int,
  or_label text,
  or_split text,
  patch_id uuid,
  PRIMARY KEY ((patch_id))
);


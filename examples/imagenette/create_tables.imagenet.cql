//////////////////////////////////////////////////////////////////////
// imagenet: https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data

CREATE KEYSPACE imagenet WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};

// Tables with original files

// Mapping patch uuid to data and label
CREATE TABLE imagenet.data_train_orig(
  patch_id uuid,
  label int,
  data blob,
  PRIMARY KEY ((patch_id))
);

// Mapping patch uuid to metadata
CREATE TABLE imagenet.metadata_train_orig(
  label int,
  or_label text,
  or_split text,
  patch_id uuid,
  PRIMARY KEY ((patch_id))
);

// Mapping patch uuid to data and label
CREATE TABLE imagenet.data_val_orig(
  patch_id uuid,
  label int,
  data blob,
  PRIMARY KEY ((patch_id))
);

// Mapping patch uuid to metadata
CREATE TABLE imagenet.metadata_val_orig(
  label int,
  or_label text,
  or_split text,
  patch_id uuid,
  PRIMARY KEY ((patch_id))
);

// Tables with center-cropped JPEGs

// Mapping patch uuid to data and label
CREATE TABLE imagenet.data_train_224_jpg(
  patch_id uuid,
  label int,
  data blob,
  PRIMARY KEY ((patch_id))
);

// Mapping patch uuid to metadata
CREATE TABLE imagenet.metadata_train_224_jpg(
  label int,
  or_label text,
  or_split text,
  patch_id uuid,
  PRIMARY KEY ((patch_id))
);

// Mapping patch uuid to data and label
CREATE TABLE imagenet.data_val_224_jpg(
  patch_id uuid,
  label int,
  data blob,
  PRIMARY KEY ((patch_id))
);

// Mapping patch uuid to metadata
CREATE TABLE imagenet.metadata_val_224_jpg(
  label int,
  or_label text,
  or_split text,
  patch_id uuid,
  PRIMARY KEY ((patch_id))
);


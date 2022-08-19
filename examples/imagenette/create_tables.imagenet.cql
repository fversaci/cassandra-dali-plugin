//////////////////////////////////////////////////////////////////////
// imagenet: https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data

CREATE KEYSPACE imagenet WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};

// Tables with original files

// Mapping patch metadata to data via uuid
CREATE TABLE imagenet.ids_orig(
  label int,
  or_label text,
  or_split text,
  patch_id uuid,
  PRIMARY KEY ((or_split, label), patch_id)
);

// Mapping patch uuid to data and label
CREATE TABLE imagenet.data_orig(
  patch_id uuid,
  label int,
  data blob,
  PRIMARY KEY ((patch_id))
);

// Mapping patch uuid to metadata
CREATE TABLE imagenet.metadata_orig(
  label int,
  or_label text,
  or_split text,
  patch_id uuid,
  PRIMARY KEY ((patch_id))
);

// Tables with center-cropped JPEGs

// Mapping patch metadata to data via uuid
CREATE TABLE imagenet.ids_224_jpg(
  label int,
  or_label text,
  or_split text,
  patch_id uuid,
  PRIMARY KEY ((or_split, label), patch_id)
);

// Mapping patch uuid to data and label
CREATE TABLE imagenet.data_224_jpg(
  patch_id uuid,
  label int,
  data blob,
  PRIMARY KEY ((patch_id))
);

// Mapping patch uuid to metadata
CREATE TABLE imagenet.metadata_224_jpg(
  label int,
  or_label text,
  or_split text,
  patch_id uuid,
  PRIMARY KEY ((patch_id))
);

// Tables with center-cropped TIFFs

// Mapping patch metadata to data via uuid
CREATE TABLE imagenet.ids_224_tiff(
  label int,
  or_label text,
  or_split text,
  patch_id uuid,
  PRIMARY KEY ((or_split, label), patch_id)
);

// Mapping patch uuid to data and label
CREATE TABLE imagenet.data_224_tiff(
  patch_id uuid,
  label int,
  data blob,
  PRIMARY KEY ((patch_id))
);

// Mapping patch uuid to metadata
CREATE TABLE imagenet.metadata_224_tiff(
  label int,
  or_label text,
  or_split text,
  patch_id uuid,
  PRIMARY KEY ((patch_id))
);


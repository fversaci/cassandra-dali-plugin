//////////////////////////////////////////////////////////////////////
// imagenette: https://github.com/fastai/imagenette

CREATE KEYSPACE imagenette WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 3};

// Tables with JPEGs

// Mapping patch metadata to data via uuid
CREATE TABLE imagenette.ids_224_jpg(
  label int,
  or_label text,
  or_split text,
  patch_id uuid,
  PRIMARY KEY ((or_split, label), patch_id)
);

// Mapping patch uuid to data and label
CREATE TABLE imagenette.data_224_jpg(
  patch_id uuid,
  label int,
  data blob,
  PRIMARY KEY ((patch_id))
);

// Mapping patch uuid to metadata
CREATE TABLE imagenette.metadata_224_jpg(
  label int,
  or_label text,
  or_split text,
  patch_id uuid,
  PRIMARY KEY ((patch_id))
);

// Tables with TIFFs (one byte per pixel/color)

// Mapping patch metadata to data via uuid
CREATE TABLE imagenette.ids_224_tiff(
  label int,
  or_label text,
  or_split text,
  patch_id uuid,
  PRIMARY KEY ((or_split, label), patch_id)
);

// Mapping patch uuid to data and label
CREATE TABLE imagenette.data_224_tiff(
  patch_id uuid,
  label int,
  data blob,
  PRIMARY KEY ((patch_id))
);

// Mapping patch uuid to metadata
CREATE TABLE imagenette.metadata_224_tiff(
  label int,
  or_label text,
  or_split text,
  patch_id uuid,
  PRIMARY KEY ((patch_id))
);

// Tables with TIFFs (one float32 per pixel/color)

// Mapping patch metadata to data via uuid
CREATE TABLE imagenette.ids_224_float(
  label int,
  or_label text,
  or_split text,
  patch_id uuid,
  PRIMARY KEY ((or_split, label), patch_id)
);

// Mapping patch uuid to data and label
CREATE TABLE imagenette.data_224_float(
  patch_id uuid,
  label int,
  data blob,
  PRIMARY KEY ((patch_id))
);

// Mapping patch uuid to metadata
CREATE TABLE imagenette.metadata_224_float(
  label int,
  or_label text,
  or_split text,
  patch_id uuid,
  PRIMARY KEY ((patch_id))
);

// Tables with original files

// Mapping patch metadata to data via uuid
CREATE TABLE imagenette.ids_orig(
  label int,
  or_label text,
  or_split text,
  patch_id uuid,
  PRIMARY KEY ((or_split, label), patch_id)
);

// Mapping patch uuid to data and label
CREATE TABLE imagenette.data_orig(
  patch_id uuid,
  label int,
  data blob,
  PRIMARY KEY ((patch_id))
);

// Mapping patch uuid to metadata
CREATE TABLE imagenette.metadata_orig(
  label int,
  or_label text,
  or_split text,
  patch_id uuid,
  PRIMARY KEY ((patch_id))
);

//////////////////////////////////////////////////////////////////////
// imagenette: https://github.com/fastai/imagenette

CREATE KEYSPACE IF NOT EXISTS imagenette WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};

// Tables with original files

// Mapping patch uuid to data and label
CREATE TABLE IF NOT EXISTS imagenette.data_train(
  id uuid,
  label int,
  data blob,
  PRIMARY KEY ((id))
) WITH compression = {'enabled':'false'};

// Mapping patch uuid to metadata
CREATE TABLE IF NOT EXISTS imagenette.metadata_train(
  label int,
  or_label text,
  or_split text,
  id uuid,
  PRIMARY KEY ((id))
);

// Mapping patch uuid to data and label
CREATE TABLE IF NOT EXISTS imagenette.data_val(
  id uuid,
  label int,
  data blob,
  PRIMARY KEY ((id))
) WITH compression = {'enabled':'false'};

// Mapping patch uuid to metadata
CREATE TABLE IF NOT EXISTS imagenette.metadata_val(
  label int,
  or_label text,
  or_split text,
  id uuid,
  PRIMARY KEY ((id))
);

# Training using a split file
  
In this example we will use the Imagenette2-320 to create a train and
validation split starting from the data stored in a Cassandra database.
We need to have Imagenette images already stored into at least a table of the db. 
To accomplish that we can follow the instructions from the section ... to the section ... of the Imagenette example.
Once the data are in the db, we can create a split file by running the ```create_split.py``` script. Running following command:

```bash
python3 create_split.py --help
```
we can see the options:
```
Usage: create_split.py [OPTIONS]

Create Split: a splitfile generator starting from data stored on a Cassandra db.

Options:  
  -k, --keyspace=STR            Specify the Cassandra keyspace   
  -s, --table-suffix=STR        Specify the table_suffix (e.g. test is the suffix for data_test and metadata_test tables)  
  
  -i, --id-col=STR              The name of the db column that contains data IDs (default: patch_id)  
  -d, --data-col=STR            The name of the db column where actual data is stored (default: data)  
  -l, --label-col=STR           db column with label (default: label)  
  -t, --label-type=STR          it is the type of the label (a string: 'int', 'image' o 'none') (default: int)  
  --metadata-ifn=STR            The input filename of previous cached metadata  
  --metadata-ofn=STR            The filename to cache  metadata read from db  
  -o, --split-ofn=STR           The name of the output splitfile  
  -r, --split-ratio=TOLIST      a comma separated values list that specifies the data proportion among desired splits (default: [8, 2])  
  -b, --balance=PARSE_BALANCE   balance configuration among classes for each split (it can be a string ('original', 'random') or a a comma separated values list with one entry for each class (default: original)  

Other actions:  
  -h, --help                    Show the help  
```

To create a training and a validation split starting from the training table of 256 sized images of imagenette, we can issue a command like:

```bash
python3 create_split.py -k imagenette -s train_256_jpg -r 8,2 -o imagenette_splitfile.pckl
```
We will got an output file with all the information to train a model with the 80% of images in the db table used as training data and the remaining 20% to validate the model. The samples selected for each split are specified in the 'split' field of the splitfile. Here an example of a generated splitfile. All information needed to gather data from the db are stored into it. 

```
{'keyspace': 'imagenette',  
 'table_suffix': 'train_256_jpg',  
 'id_col': 'patch_id',  
 'data_col': 'data',  
 'label_type': 'int',  
 'label_col': 'label',  
 'row_keys': array([UUID('fc55f4c0-22a0-45d3-b551-a0c32d39b2c9'),  
        UUID('893dce7d-2fd1-4272-b069-0b1b9542adc7'),  
        UUID('c4f0c362-ca78-4863-8e92-d7d33d2a67de'), ...,  
        UUID('98d0c32f-d860-4fd4-a4b8-a472c28dc98e'),  
        UUID('5c3139fe-ac20-4365-83f5-a532f9050947'),  
        UUID('7801dc6c-dfc8-4d20-b0a6-af8687169b3f')], dtype=object),  
 'split': [array([2704, 7617, 6172, ..., 6009, 8467, 7782]),  
  array([1329, 6671,  978, ..., 2769, 4708, 2198])],  
 'num_classes': 10}  
```

To avoid reading metadata from the database everytime we need to create a new split we can save metadata to a file specifing its name by using the CLI option ```--metadata-ofn```. For instance issuing:
 
```bash
python3 create_split.py -k imagenette -s train_256_jpg -r 8,2 --metadata-ofn metadata.cache -o imagenette_splitfile.pckl
```

The next time we want generate a new split we can skip passing db information by using the CLI option ```--metadata-ifn``` to pass the filename of the file with cached metadata:

```bash
python3 create_split.py --metadata-ifn metadata.cache -r 8,2 -o imagenette_splitfile.pckl
```

If the metadata table is large in size, caching it to a file can save time for creating new splits.


## Multi-GPU training using the split file

To train and validate a model with the generated split file, just run:

```bash
$ torchrun --nproc_per_node=1 distrib_train_from_cassandra.py --split-fn prova_split.pckl -a resnet50 --dali_cpu --b 128 --loss-scale 128.0 --workers 4 --lr=0.4 --opt-level O2
```

The split file passed by the required ```--split-fn``` option provides all the information to get the right taining and validation samples from the db. 

Training and validation split specified by the ```--train-index``` and ```val-index``` CLI options. Their default values are 0 and 1 respectively meaning that the row 0 of the array in the ```split``` field of the splitfile is used as training dataset whereas the row 1 is used for the validation step.

So, if we create a split with:

```bash
python3 create_split.py --metadata-ifn metadata.cache -r 2,8 -o imagenette_splitfile.pckl
```

with the first split containing the 20% of the db table data, probably we want to specify the training and validation index as follow:


```bash
$ torchrun --nproc_per_node=1 distrib_train_from_cassandra.py --split-fn prova_split.pckl --train-index 1 --val-index 0 -a resnet50 --dali_cpu --b 128 --loss-scale 128.0 --workers 4 --lr=0.4 --opt-level O2
```

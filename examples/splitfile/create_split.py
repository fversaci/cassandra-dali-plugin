from clize import run, parser 
from split_generator import imagenet_split_generator

@parser.value_converter
def tolist(arg):
    try:
        return list(eval(arg))
    except:
        raise ValueError(f"{arg}... Please insert comma separated values")

@parser.value_converter
def parse_balance(arg):
    if len(arg.split(',')) > 1:
        return tolist(arg)
    else:
        return arg

def main(*, keyspace:'k'=None, table_suffix:'s'=None, id_col:'i'='patch_id', data_col:'d'='data', label_col:'l'='label', label_type:'t'='int', 
        metadata_ifn=None, metadata_ofn=None, split_ofn:'o'=None, 
        split_ratio:('r', tolist)=[8,2], balance:('b', parse_balance)='original'):
    """
    Create Split: a splitfile generator starting from data stored on a Cassandra db.

    :param keyspace: Specify the Cassandra keyspace
    :param table_suffix: Specify the table_suffix (e.g. test is the suffix for data_test and metadata_test tables)
    :param id_col: The name of the db column that contains data IDs 
    :param data_col: The name of the db column where actual data is stored 
    :param label_col: db column with label 
    :param label_type: it is the type of the label (a string: 'int', 'image' o 'none')
    :param metadata_ifn: The input filename of previous cached metadata 
    :param metadata_ofn: The filename to cache  metadata read from db
    :param split_ofn: The name of the output splitfile 
    :param split_ratio: a comma separated values list that specifies the data proportion among desired splits
    :param balance: balance configuration among classes for each split (it can be a string ('original', 'random') or a a comma separated values list with one entry for each class
    """

    isg = imagenet_split_generator()
    
    # Load metadata
    if not metadata_ifn:
        print ("Loading metadata from database")
        from private_data import CassConf as CC
        isg.load_from_db(CC, keyspace, table_suffix)
    else:
        print("Loading metadata from file")
        isg.load_from_file(metadata_ifn)

    # Metadata saving
    if metadata_ofn:
        print(f"Saving metadata dataframe to {metadata_ofn}")
        isg.cache_db_data_to_file(metadata_ofn)
    
    print (f"Creating {len(split_ratio)} splits")    
    isg.create_split(split_ratio, balance=balance)

    if split_ofn:
        print(f"Saving splitfile: {split_ofn}")
        isg.save_splits(split_ofn)

if __name__ == '__main__':
    run(main)

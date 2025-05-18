import pickle
import polars as pl
import json

with open('/home/yaoyi/pyo00005/Mapping_Prejudice/logs/regular/anoka.pkl', 'rb') as handle:
    data = pickle.load(handle)

def collect_terms(str_basic, str_fuzzy):
    dict_basic = {}
    dict_fuzzy = {}

    try: 
        dict_basic = json.loads(str_basic)
        del dict_basic['workflow']
        del dict_basic['lookup']
        del dict_basic['uuid']
    except: pass

    try: 
        dict_fuzzy = json.loads(str_fuzzy)
        del dict_fuzzy['workflow']
        del dict_fuzzy['lookup']
        del dict_fuzzy['uuid']
    except: pass

    # merge dict basic and dict fuzzy, remain only those that are unique keys.
    dict_basic.update(dict_fuzzy)

    dict_without = dict_basic.copy()
    try:
        del dict_without['nationality']
    except: pass

    # create two version, one with nationality other without nationality
    return {'with': list(dict_basic.keys()), 'without': list(dict_without.keys())}
    # return 0

# pl_original_data = pl.read_csv('/home/yaoyi/pyo00005/Mapping_Prejudice/ground_truth/zooniverse/splitted_data/mn-dakota/test.csv', infer_schema_length=0)
pl_original_data = pl.read_csv('/home/yaoyi/pyo00005/Mapping_Prejudice/ground_truth/zooniverse/labeled_data/mn-anoka-county_deedpage_sample_post_zooniverse_mn-anoka-county_100pct_20250410_1706.csv', infer_schema_length=0)
pl_original_data = pl_original_data.select(
    pl.col(['page_ocr_text', 'hit_contents_basic', 'hit_contents_fuzzy'])
)
pl_original_data = pl_original_data.with_columns(
    tmp = pl.struct(pl.all()).map_elements(lambda x: collect_terms(x['hit_contents_basic'], x['hit_contents_fuzzy']))
).unnest('tmp')

data = data.with_columns(
    token_count = pl.col('prefix_tags').str.split(" ").list.len()
).select(
    pl.col(['page_ocr_text', 'ner_identified', 'token_count', 'sentence'])
).group_by('page_ocr_text').agg([pl.all()]).with_columns(
    pl.col('token_count').list.sum(),
    pl.col('ner_identified').map_elements(lambda x: [item for sublist in x for item in sublist])
)

data = data.join(pl_original_data, on='page_ocr_text', how='left')

data = data.with_columns(
    len_identified = (pl.col('ner_identified').list.len())>0,
    len_with = (pl.col('with').list.len())>0,
    len_without = (pl.col('without').list.len())>0,
).filter(
    pl.col('len_identified') == False,
    pl.col('len_without') == True
)

# pl_full = pl.concat(
#     [data, pl_original_data],
#     how='align'
# )

print(data.shape[0])
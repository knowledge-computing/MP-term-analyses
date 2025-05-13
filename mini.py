import pickle
import polars as pl

def split_str_list(str_list, type_num:bool=False):
    str_list = str_list.strip("['']").replace('\n', '')
    
    if type_num:
        list_list = str_list.split(" ")
    else:
        list_list = str_list.split("' '")

    return list_list

def group_classify(bool_gt, bool_identified):
    if bool_gt and bool_identified:
        return 'TP'
    if (not bool_gt) and (not bool_identified):
        return 'TN'
    
    if (not bool_gt) and bool_identified:
        return 'FP'
    
    if bool_gt and (not bool_identified):
        return 'FN'

with open('/home/yaoyi/pyo00005/Mapping_Prejudice/logs/regular/washington.pkl', 'rb') as handle:
    pl_data = pickle.load(handle)

pl_data = pl_data.with_columns(
    pl.col('tokens').map_elements(lambda x: split_str_list(x)),
    pl.col('ner_tags').map_elements(lambda x: split_str_list(x)),
    pl.col('prefix_tags').map_elements(lambda x: split_str_list(x, True)),
)

pl_data = pl_data.with_columns(
    ground_truth = pl.col('prefix_tags').list.contains('1'),
    identified = (pl.col('ner_identified').list.len() > 0),
    total_tokens = pl.col('prefix_tags').list.len()
).with_columns(
    output_type = pl.struct(pl.all()).map_elements(lambda x: group_classify(x['ground_truth'], x['identified']))
)

print(pl_data.select(pl.sum('total_tokens')))

print('TP: ', pl_data.filter(pl.col('output_type') == 'TP').shape[0], '\tTN: ',  pl_data.filter(pl.col('output_type') == 'TN').shape[0], '\nFN: ',  pl_data.filter(pl.col('output_type') == 'FN').shape[0],'\tFP: ', pl_data.filter(pl.col('output_type') == 'FP').shape[0])
print(pl_data)

pl_neg = pl_data.with_columns(
    pl.col('ner_tags').list.unique()
).filter(pl.col('ner_tags').list.len() == 1,
         pl.col('ner_tags').list.first() == 'O')

# pl_neg = pl_data.filter(pl.col('tokens').list.unique() == 0)
# true_neg = pl_neg.filter(pl.col('ner_identified').list.len() == 0)
false_neg = pl_neg.filter(pl.col('ner_identified').list.len() != 0)
print(f"FP: {false_neg.shape[0]}")


false_neg.to_pandas().to_csv('./falseneg.csv')



def get_true_pos(list_gt:list, list_predicted:list):
    len_gt = len(list_gt)
    len_match = (len(set(list_gt) & set(list_predicted)))

    if len_gt == len_match:
        return len_gt

    if list_gt == list_predicted:
        return len_gt
    
    if (len_gt == 1) and (len(list_predicted) == 1):
        if list_gt[0] in list_predicted[0]:
            return 1

    if len_match > 0:
        return len_match

    # if len_gt != len_match:
    #     print(list_gt, list_predicted)

    return 0

def check_positive_match(list_bool_tags, list_tokens, list_ners):
    # print(list_bool_tags, list_tokens, list_ners)
    ner_true = []
    if len(list_bool_tags) == len(list_tokens):
        for idx, i in enumerate(list_bool_tags):
            if i != 'O':
                if (list_tokens[idx] == 'domestic') and ('servant' in list_tokens[idx+1]):
                    ner_true.append(f'{list_tokens[idx]} {list_tokens[idx + 1]}')
                elif (list_tokens[idx] != 'servants'):
                    ner_true.append(list_tokens[idx])
    
    elif len(list_bool_tags) == len(list_tokens) + 2:
        for idx, i in enumerate(list_bool_tags):
            if i != 'O':
                try:
                    if (list_tokens[idx] == 'domestic') and ('servant' in list_tokens[idx+1]):
                        ner_true.append(f'{list_tokens[idx-2]} {list_tokens[idx - 1]}')
                    elif (list_tokens[idx] != 'servants'):
                        ner_true.append(list_tokens[idx-2])
                except:
                    ner_true.append((list_tokens[idx-2]))

                # ner_true.append(list_tokens[idx-2])
                    
    else: 
        for idx, i in enumerate(list_bool_tags):
            if i != 'O':
                ner_true.append(list_tokens[idx])
            
    if ner_true == list_ners:
        return {'true_count': len(ner_true), 'match_count': len(ner_true)}
    
    # print("phase 2 pass")

    count = 0
    if len(ner_true) == len(list_ners):
        for i in list(range(len(ner_true))):
            if (ner_true[i] in list_ners[i]) or (list_ners[i] in ner_true[i]):
                count += 1
        
        return {'true_count': len(ner_true), 'match_count': count}
    
    # print("phase 3 pass")

    print(ner_true, list_ners)

    return {'true_count': len(ner_true), 'match_count': len(set(ner_true) and set(list_ners))}

pl_pos = pl_data.filter(pl.col('ner_tags').list.unique().list.len() > 1)
pl_pos = pl_pos.with_columns(
    tmp = pl.struct(pl.all()).map_elements(lambda x: check_positive_match(x['ner_tags'], x['tokens'], x['ner_identified']))
).unnest('tmp')

print('TP:', pl_pos.select(pl.sum('match_count')).item(0, 'match_count'))

pl_pos = pl_pos.filter(pl.col('match_count') == 0)

pl_pos.to_pandas().to_csv('./falsepos.csv')

pl_pos = pl_pos.select(
    pl.col(['page_ocr_text','sentence','type','tokens','ner_tags','prefix_tags','ner_identified','true_count'])
)

# pl_pos = pl_data.filter(pl.col('tokens').list.len() != 0)
# pl_pos = pl_pos.with_columns(
#     test = pl.struct(pl.all()).map_elements(lambda x: get_true_pos(x['tokens'], x['ner_identified']))
# )

# print(pl_pos.sort('test').item(200, 'text'))

# tp_count = pl_pos.select(pl.sum("test")).item(0, 'test')
# print(f"TP: {tp_count}")

# fn_count = pl_pos.explode('tokens').shape[0] - tp_count
# print(f"FN: {fn_count}")

# fn_not_racial = pl_pos.explode('tokens').filter(pl.col('tokens') != 'racial').shape[0] - tp_count
# print(f"FN (exluding term racial): {fn_not_racial}")

# pl_not = pl_pos.filter(pl.col('test') == 0)
# pd_not = pl_not.to_pandas()

# # pd_not.to_csv('./check.csv')


# Importing gc module
import gc
 
# Returns the number of
# objects it has collected
# and deallocated
collected = gc.collect()
 
# Prints Garbage collector 
# as 0 object
print("Garbage collector: collected",
          "%d objects." % collected)
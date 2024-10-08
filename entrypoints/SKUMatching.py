import os
import importlib
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from utils.ImageFuncs import *
from constants.FileNameConstants import *
from utils.DataEnrichmentHelper import *
from utils.GeneralHelper import *
from utils.SpellCorrectHelper import *
from utils.FeatureExtractionHelper import *
from utils.MatchingLogicHelper import *
from utils.MatchingAuxHelper import *
from utils.DataFetchHelper import *
from utils.ConfigHelper import *
# from utils.Initialize import Initialize
from utils.Slack import *
from utils.SlackDecorator import *
from constants.SlackConstants import SlackConstants
from utils.decorators import error_decorator
Slack_=Slack(SlackConstants.bot_token)
# Initialize().download_config()
from utils.log import Log
import time
import json
from tqdm import tqdm
log = Log(__name__)



class SKUMatching:

    def __init__(self, client_name, run_date, max_n):
        self.client_name = client_name
        self.run_date = run_date
        self.ensure_folders()
        if max_n is None:
            self.max_n= 50
        else:
            self.max_n = int(max_n)
        config_helper = ConfigHelper(self.client_name)
        created_params = config_helper.create_feature_params()
        for k,v in created_params.items():
            self.params[k] = v
        try:
            unmatched_total_path = os.path.join(self.params['dirs']['processed_files_folder'], f'{client_name}_processed_mismatches.parquet')
            s3_path = get_s3_path(unmatched_total_path, self.client_name, self.run_date, processed=True)
            if os.path.exists(unmatched_total_path)==False and s3_object_exists(s3_path):
                s3_download_file(s3_path, unmatched_total_path)
            df_um_tot = pd.read_parquet(unmatched_total_path)
            df_um_tot['ocr_res'] = df_um_tot['ocr_res'].apply(lambda x: ast.literal_eval(x) if x is not None else x)
            df_um_tot['ner_out_2'] = df_um_tot['ner_out_2'].apply(lambda x: ast.literal_eval(x) if x is not None else x)
            df_um_tot['ner_out'] = df_um_tot['ner_out'].apply(lambda x: ast.literal_eval(x) if x is not None else x)
            df_um_tot = df_um_tot.loc[df_um_tot['product_name']!='NONE',:].reset_index(drop=True)
            for feature in self.params['llama_features']:
                extracted_feat_ls = []
                for raw_feat in df_um_tot[f'llm_{feature}'].values:
                    extracted_feat_ls.append(extract_feature_for_llama(raw_feat, feature))
                df_um_tot[f'{feature}_llm'] = extracted_feat_ls
                df_um_tot[f'{feature}_llm'] = df_um_tot[f'{feature}_llm'].apply(lambda x: ' '.join([y for y in x if y!='None']))
                df_um_tot.loc[df_um_tot[f'{feature}_llm']=='', f'{feature}_llm'] = 'None'
                df_um_tot[f'{feature}_check'] = np.vectorize(check_feature_in_title)(df_um_tot['product_name'], df_um_tot[f'{feature}_llm'])
                df_um_tot[f'{feature}_llm_orig'] = df_um_tot[f'{feature}_llm']
                df_um_tot.loc[df_um_tot[f'{feature}_check']==False, f'{feature}_llm'] = 'None'
            ## ============================deprecated =============================================  
            #for feature in self.params['uni_ner_features']:
                #feature_orig = feature
                #df_um_tot[f'{feature}_uni_ner']= df_um_tot[f'uni_ner_{feature}'].apply(lambda x: x[0] if len(x)>0 else '')
            ## ====================================================================================
            if len(self.params['llama_features'])>0:
                ## ============================deprecated =========================================  
                #df_um_tot = self.combine_common_extracted_features(df_um_tot)
                ## ================================================================================
                for feature in self.params['llama_features']:
                    feature_orig = feature
                    df_um_tot[feature_orig] = df_um_tot[f'{feature}_llm']
            df_um_tot['sku_priority'] = ''
            df_um_tot.loc[(df_um_tot['image_count']>0),  'sku_priority'] = 'Recent & Full Data'
            df_um_tot.loc[(df_um_tot['image_count']==0), 'sku_priority'] = 'Recent & No Image'

            self.df_um_tot = df_um_tot
            log.info(f"Unmatched Data Shape:  {df_um_tot.shape}")
        except Exception as e:
            log.info('***** Error reading Unmatched Data File *****')
            log.info(e)
            self.df_um_tot = None
            #raise Exception('SKU Data File does not exist')
        try:
            
            pim_path = os.path.join(self.params['dirs']['processed_files_folder'], f'{client_name}_processed_pim.parquet')
            s3_path = get_s3_path(pim_path, self.client_name, self.run_date, processed=True)
            if os.path.exists(pim_path)==False and s3_object_exists(s3_path):
                s3_download_file(s3_path, pim_path)
            df_pim = pd.read_parquet(pim_path)
            df_pim['ocr_res'] = df_pim['ocr_res'].apply(lambda x: ast.literal_eval(x) if x is not None else x)
            df_pim['ner_out_2'] = df_pim['ner_out_2'].apply(lambda x: ast.literal_eval(x) if x is not None else x)
            df_pim['ner_out'] = df_pim['ner_out'].apply(lambda x: ast.literal_eval(x) if x is not None else x)
            df_pim = df_pim.loc[df_pim['product_name']!='NONE',:].reset_index(drop=True)
            
            for feature in self.params['llama_features']:
                extracted_feat_ls = []
                for raw_feat in df_pim[f'llm_{feature}'].values:
                    extracted_feat_ls.append(extract_feature_for_llama(raw_feat, feature))
                df_pim[f'{feature}_llm'] = extracted_feat_ls
                df_pim[f'{feature}_llm'] = df_pim[f'{feature}_llm'].apply(lambda x: ' '.join([y for y in x if y!='None']))
                df_pim.loc[df_pim[f'{feature}_llm']=='', f'{feature}_llm'] = 'None'
                df_pim[f'{feature}_check'] = np.vectorize(check_feature_in_title)(df_pim['product_name'], df_pim[f'{feature}_llm'])
                df_pim[f'{feature}_llm_orig'] = df_pim[f'{feature}_llm']
                df_pim.loc[df_pim[f'{feature}_check']==False, f'{feature}_llm'] = 'None'
            ## ============================deprecated =============================================    
            #for feature in self.params['uni_ner_features']:
            #   feature_orig = feature
            #    df_pim[f'{feature}_uni_ner']= df_pim[f'uni_ner_{feature}'].apply(lambda x: x[0] if len(x)>0 else '')
            if len(self.params['llama_features'])>0:
                #df_pim = self.combine_common_extracted_features(df_pim)
                for feature in self.params['llama_features']:
                    feature_orig = feature
                    df_pim[feature_orig] = df_pim[f'{feature}_llm']
            self.df_pim = df_pim
            log.info(f"PIM Data Shape: {df_pim.shape}")
        except Exception as e:
            log.info('***** Error reading PIM Data File *****')
            log.info(e)
            self.df_pim = None
            #raise Exception('PIM File does not exist')
            
    def combine_common_extracted_features(self, df):
        common_features = set(self.params['uni_ner_features']).intersection(set(self.params['llama_features']))
        
        for feature in common_features:

            df.loc[df[f'{feature}_uni_ner']=='', f'{feature}_uni_ner'] =  df.loc[df[f'{feature}_uni_ner']=='', f'{feature}_llm']
            df.loc[df[f'{feature}_llm']=='', f'{feature}_llm'] =  df.loc[df[f'{feature}_llm']=='', f'{feature}_uni_ner']
            
        return df   
        
    def beautify_brand(self, df_flt):
        df_flt['Candidate GTIN Metadata'] = df_flt['Candidate GTIN Metadata'].apply(lambda x: ast.literal_eval(str(x)) )
        for i in tqdm(range(0, len(df_flt))):
            if df_flt.loc[i, 'Brand_change_flag']==1:
                sku_brand = df_flt.loc[i, 'Retailer SKU Metadata']['auxiliary_data']['Brand']
                df_flt.loc[i, 'Candidate GTIN Metadata']['auxiliary_data']['Brand']= sku_brand
        return df_flt
    def ensure_folders(self):
        params = dict()
        params['dirs'] = dict()   
        params['dirs']['base_folder'] = f"{BASE_FOLDER}/"
        if not os.path.exists(BASE_FOLDER):
            os.makedirs(BASE_FOLDER)
            
        client_folder = os.path.join(BASE_FOLDER,self.client_name)
        params['dirs']['client_folder'] = client_folder
        if not os.path.exists(client_folder):
            os.makedirs(client_folder)
            
        raw_data_folder = os.path.join(BASE_FOLDER,self.client_name, RAW_DATA_FOLDER, self.run_date.split("-")[0], self.run_date.split("-")[1], self.run_date.split("-")[2])
        params['dirs']['raw_data_folder'] = raw_data_folder
        if not os.path.exists(raw_data_folder):
            os.makedirs(raw_data_folder)
            
        cached_folder = os.path.join(BASE_FOLDER, self.client_name, CACHED_FOLDER, self.run_date.split("-")[0], self.run_date.split("-")[1], self.run_date.split("-")[2])
        params['dirs']['cached_files_folder'] = cached_folder
        if not os.path.exists(cached_folder):
            os.makedirs(cached_folder)
            
        cached_ocr_folder = os.path.join(BASE_FOLDER, self.client_name, OCR_FOLDER)
        params['dirs']['cached_ocr_files_folder'] = cached_ocr_folder
        if not os.path.exists(cached_ocr_folder):
            os.makedirs(cached_ocr_folder)
            
        processed_folder = os.path.join(BASE_FOLDER, self.client_name, PROCESSED_FOLDER,  self.run_date.split("-")[0], self.run_date.split("-")[1], self.run_date.split("-")[2])
        params['dirs']['processed_files_folder'] = processed_folder
        if not os.path.exists(processed_folder):
            os.makedirs(processed_folder)
        self.params = params
        
    def create_text_corpus(self):
        text_coprus_path = os.path.join(BASE_FOLDER, self.client_name, OCR_FOLDER, f'{self.client_name}_text.txt')
        self.params['text_coprus_path'] = text_coprus_path
        file = open(text_coprus_path,'w')
        back_mapping_dic = {}
        corpus = set()
        for item in tqdm(self.df_um_tot['name_cleaned']):
            item = remove_special_characters(item)
            item = (' '.join([x for x in item.split(' ') if x.isalpha() and len(x)>1])).lower()
            for k in item.split(' '):
                corpus.add(k)
            i1 = create_backmapping(item,1, back_mapping_dic)
            i2 = create_backmapping(item,2, back_mapping_dic)
            i3 = create_backmapping(item,3, back_mapping_dic)
            i4 = create_backmapping(item,4, back_mapping_dic)
            for st in i1.split(' '):
                file.write(st+"\n")
            for st in i2.split(' '):
                file.write(st+"\n")
            for st in i3.split(' '):
                file.write(st+"\n")
            for st in i4.split(' '):
                file.write(st+"\n")


        for item in tqdm(self.df_pim['name_cleaned']):
            item = remove_special_characters(item)
            item = (' '.join([x for x in item.split(' ') if x.isalpha() and len(x)>1])).lower()
            for k in item.split(' '):
                corpus.add(k)
            i1 = create_backmapping(item,1, back_mapping_dic)
            i2 = create_backmapping(item,2, back_mapping_dic)
            i3 = create_backmapping(item,3, back_mapping_dic)
            i4 = create_backmapping(item,4, back_mapping_dic)
            for st in i1.split(' '):
                file.write(st+"\n")
            for st in i2.split(' '):
                file.write(st+"\n")
            for st in i3.split(' '):
                file.write(st+"\n")
            for st in i4.split(' '):
                file.write(st+"\n")
                
        file.close()
        self.back_mapping_dic = back_mapping_dic
    @error_decorator
    @slack_decorator
    def run(self):
        max_n = self.max_n
        client_name = self.client_name
        run_date = self.run_date
        log.info('***** Check if the files exist *****')
        if self.df_um_tot is None:
            raise Exception('SKU Data File does not exist')
        if self.df_pim is None:
            raise Exception('PIM Data File does not exist')
        
        log.info('***** Creating Text Correction module for OCR *****')
        self.create_text_corpus()
        sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        sym_spell.create_dictionary(self.params['text_coprus_path'])

        
        log.info('***** Creating basic id level dictionaries *****')
        df_um_tot = self.df_um_tot.copy()
        original_cols_sku = list(df_um_tot.columns)
        df_pim = self.df_pim.copy()
        original_cols_pim = list(df_pim.columns)
        sku_id_to_img_path = {x:y for x, y in zip(df_um_tot.loc[df_um_tot['fail']==0, 'id'],df_um_tot.loc[df_um_tot['fail']==0, 'image_path']) }
        sku_id_to_brand = get_brand_mapping(df_um_tot, mode = 'sku')
        sku_id_to_ocr = {x:y if y is not None else [] for x, y in zip(df_um_tot['id'],df_um_tot['ocr_res'])}
        sku_id_to_ocr_corrected = get_corrected_ocr_mapping_dic(sku_id_to_ocr, self.back_mapping_dic, sym_spell)
        df_um_tot['ocr_res_corrected'] = df_um_tot['id'].apply(lambda x: sku_id_to_ocr_corrected[x])
        
        pim_id_to_img_path = {x:y for x, y in zip(df_pim.loc[df_pim['fail']==0, 'id'],df_pim.loc[df_pim['fail']==0, 'image_path']) }
        pim_id_to_brand = get_brand_mapping(df_pim, mode = 'pim')
        pim_id_to_ocr = {x:y if y is not None else [] for x, y in zip(df_pim['id'],df_pim['ocr_res'])}
        pim_id_to_ocr_corrected = get_corrected_ocr_mapping_dic(pim_id_to_ocr, self.back_mapping_dic, sym_spell)
        df_pim['ocr_res_corrected'] = df_pim['id'].apply(lambda x: pim_id_to_ocr_corrected[x])
        
        module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"     
        text_emb_model = hub.load(module_url)
        log.info(f"module {module_url} loaded")
        
        log.info('***** Creating Title Embeddings *****')
        gtin_title_emb = np.array(text_emb_model(df_pim['product_name'].str.lower()))
        gtin_title_cleaned_emb = np.array(text_emb_model(df_pim['name_cleaned'].str.lower()))

        pim_id_to_title = {}
        for i in range(df_pim.shape[0]):
            pim_id_to_title[df_pim.loc[i, 'id']] = df_pim.loc[i, 'product_name']
        pim_id_to_clean_title = {}
        for i in range(df_pim.shape[0]):
            pim_id_to_clean_title[df_pim.loc[i, 'id']] = df_pim.loc[i, 'name_cleaned']
        pim_id_to_ner_out = {}
        for i in range(df_pim.shape[0]):
            pim_id_to_ner_out[df_pim.loc[i, 'id']] = df_pim.loc[i, 'ner_out']
        pim_id_dic = {}
        pim_id_inv_dic = {}
        for i in range(df_pim.shape[0]):
            pim_id_dic[df_pim.loc[i, 'id']] = i
            pim_id_inv_dic[i] = df_pim.loc[i, 'id']

        sku_title_emb = np.array(text_emb_model(df_um_tot['product_name'].str.lower()))
        sku_title_cleaned_emb = np.array(text_emb_model(df_um_tot['name_cleaned'].str.lower()))
        df_um_tot['name_cleaned'] = df_um_tot['name_cleaned'].apply(lambda x: re.sub(r'[^\w\s]', '', x).strip())

        sku_id_to_title = {}
        for i in range(df_um_tot.shape[0]):
            sku_id_to_title[df_um_tot.loc[i, 'id']] = df_um_tot.loc[i, 'product_name']
        sku_id_to_clean_title = {}
        for i in range(df_um_tot.shape[0]):
            sku_id_to_clean_title[df_um_tot.loc[i, 'id']] = df_um_tot.loc[i, 'name_cleaned']
        sku_id_to_ner_out = {}
        for i in range(df_um_tot.shape[0]):
            sku_id_to_ner_out[df_um_tot.loc[i, 'id']] = df_um_tot.loc[i, 'ner_out']
        sku_id_dic = {}
        sku_id_inv_dic = {}
        for i in range(df_um_tot.shape[0]):
            sku_id_dic[df_um_tot.loc[i, 'id']] = i
            sku_id_inv_dic[i] = df_um_tot.loc[i, 'id']

        # download image embeddings file from S3
        log.info('*****')
        log.info('..... Downloading Image embeddings from S3 .....')
        pim_img_path_raw = os.path.join(self.params['dirs']['processed_files_folder'], f'{client_name}_pim_raw_emb.parquet')
        s3_path = get_s3_path(pim_img_path_raw, client_name, run_date, processed=True)
        if s3_download_file(s3_path, pim_img_path_raw) == 1:
            log.info('***** Downloaded PIM RAW embeddings *****')
        
        pim_img_path_pro = os.path.join(self.params['dirs']['processed_files_folder'], f'{client_name}_pim_pro_emb.parquet')
        s3_path = get_s3_path(pim_img_path_pro, client_name, run_date, processed=True)
        if s3_download_file(s3_path, pim_img_path_pro) == 1:
            log.info('***** Downloaded PIM PRO embeddings *****')
        
        sku_img_path_raw = os.path.join(self.params['dirs']['processed_files_folder'], f'{client_name}_sku_raw_emb.parquet')
        s3_path = get_s3_path(sku_img_path_raw, client_name, run_date, processed=True)
        if s3_download_file(s3_path, sku_img_path_raw) == 1:
            log.info('***** Downloaded SKU RAW embeddings *****')
        
        sku_img_path_pro = os.path.join(self.params['dirs']['processed_files_folder'], f'{client_name}_sku_pro_emb.parquet')
        s3_path = get_s3_path(sku_img_path_pro, client_name, run_date, processed=True)
        if s3_download_file(s3_path, sku_img_path_pro) == 1:
            log.info('***** Downloaded SKU PRO embeddings *****')
        
        ### Load image embeddings for PIM
        pim_dat_img_emb, pim_dat_img_id_dic, pim_dat_img_id_inv_dic, pim_dat_img_emb_dic = load_embeddings(self.params['dirs']['processed_files_folder'], client_name, file_type = 'pim', emb_type = 'raw')
        pim_dat_img_emb2, pim_dat_img_id_dic, pim_dat_img_id_inv_dic, pim_dat_img_emb_dic2 = load_embeddings(self.params['dirs']['processed_files_folder'], client_name, file_type = 'pim', emb_type = 'pro')

        ### Load image embeddings for unmatched data
        um_dat_img_emb, um_dat_img_id_dic, um_dat_img_id_inv_dic, um_dat_img_emb_dic = load_embeddings(self.params['dirs']['processed_files_folder'], client_name, file_type = 'sku', emb_type = 'raw')
        um_dat_img_emb2, um_dat_img_id_dic, um_dat_img_id_inv_dic, um_dat_img_emb_dic2 = load_embeddings(self.params['dirs']['processed_files_folder'], client_name, file_type = 'sku', emb_type = 'pro')


        ### Create Brand Corpus
        brand_corpus = create_brand_corpus(sku_id_to_brand, pim_id_to_brand)
       
        ### Create exhaustive list of vallid LLM feature classes
        llm_based_feature_ls = {}
        for feature in self.params['llama_features']:
            feature_orig = feature
            stop_words = self.params['stop_words'].get(feature, [])
            feat_dic = {}
            feat_dic = create_feature_dic(df_pim, feat_dic, feature, pim_id_to_brand)
            feat_dic = create_feature_dic(df_um_tot, feat_dic, feature, sku_id_to_brand)
            feat_dic = {k: v for k, v in sorted(feat_dic.items(), key=lambda item: -1 * item[1])}
            feat_ls = list(feat_dic.keys())
            feat_stop_words = stop_words + brand_corpus
            for kw in feat_ls:
                if kw in feat_stop_words:
                    feat_ls.remove(kw)
            llm_based_feature_ls[feature_orig] = feat_ls
        
        
        ### Classify features 
        attribute_kw_dic = self.params['attribute_kw_dic']
        attribute_kw_dic_ocr = self.params['attribute_kw_dic_ocr']
        
        df_um_tot = classify_attribute(df_um_tot, attribute_kw_dic)
        # add when OCR boxing Logic is correct
        # df_um_tot =classify_attribute_ocr(df_um_tot, attribute_kw_dic_ocr)
        
        df_pim = classify_attribute(df_pim, attribute_kw_dic)
        # add when OCR boxing Logic is correct
        # df_pim =classify_attribute_ocr(df_pim, attribute_kw_dic_ocr)

        attribute_kw_dic_bool = self.params['attribute_kw_dic_bool']
        attribute_kw_dic_ocr_bool = self.params['attribute_kw_dic_ocr_bool']
        for bool_feat, classification_val_dic in attribute_kw_dic_bool.items():

            class_val = list(classification_val_dic.keys())[0]
            sku_f_txt = df_um_tot['product_name'].apply(lambda x: classify_attribute_util(x, classification_val_dic)==class_val)
            pim_f_txt = df_pim['product_name'].apply(lambda x: classify_attribute_util(x, classification_val_dic)==class_val)

            sku_f_ocr = df_um_tot['ocr_res_corrected'].apply(lambda x: is_attribute_based_ocr(x, attribute_kw_dic_ocr_bool[bool_feat])[class_val])
            pim_f_ocr = df_pim['ocr_res_corrected'].apply(lambda x: is_attribute_based_ocr(x, attribute_kw_dic_ocr_bool[bool_feat])[class_val])

            df_um_tot[bool_feat] = np.vectorize(compute_boolean_or)(sku_f_txt, sku_f_ocr)
            df_pim[bool_feat] = np.vectorize(compute_boolean_or)(pim_f_txt, pim_f_ocr)
        new_cols_sku = set(df_um_tot.columns) - set(original_cols_sku)
        new_cols_pim = set(df_pim.columns) - set(original_cols_pim)
        log.info("*****")
        log.info(f"Added columns: {new_cols_sku} to SKU data")
        log.info(f"Added columns: {new_cols_pim} to PIM data")
        log.info("*****")
        
        log.info('***** Creating Auxillary feature mapping on ID level *****')
        aux_features = {}

        for feat in self.params['kw_features']:
            aux_features[feat] = 'c'
        for feat in self.params['llama_features']:
            aux_features[feat] = 'k'

        sku_id_to_aux_feat = create_id_to_aux_feat(df_um_tot, aux_features, sku_id_to_brand, sku_id_to_ocr_corrected, llm_based_feature_ls)
        sku_id_to_aux_feat = add_mandatory_features(self.params, sku_id_to_aux_feat, sku_id_to_ner_out, sku_id_to_brand)
        
        pim_id_to_aux_feat = create_id_to_aux_feat(df_pim, aux_features, pim_id_to_brand, pim_id_to_ocr_corrected, llm_based_feature_ls)
        pim_id_to_aux_feat = add_mandatory_features(self.params,pim_id_to_aux_feat, pim_id_to_ner_out, pim_id_to_brand)
        
        sim_mat_text1 =  np.inner(sku_title_emb, gtin_title_emb)
        sim_mat_text2 =  np.inner(sku_title_emb, gtin_title_cleaned_emb)
        sim_mat_text3 =  np.inner(sku_title_cleaned_emb, gtin_title_emb)
        sim_mat_text4 =  np.inner(sku_title_cleaned_emb, gtin_title_cleaned_emb)

        if len(pim_dat_img_emb)>0 and len(um_dat_img_emb)>0:
            sim_mat_img1 =  np.inner(um_dat_img_emb, pim_dat_img_emb)
            sim_mat_img2 =  np.inner(um_dat_img_emb, pim_dat_img_emb2)
            sim_mat_img3 =  np.inner(um_dat_img_emb2, pim_dat_img_emb)
            sim_mat_img4 =  np.inner(um_dat_img_emb2, pim_dat_img_emb2)
        else:
            sim_mat_img1 =  None
            sim_mat_img2 =  None
            sim_mat_img3 =  None
            sim_mat_img4 =  None

        pim_dics = {
                    'pim_id_dic' : pim_id_dic, 
                    'pim_id_inv_dic':pim_id_inv_dic,
                    'pim_dat_img_id_dic':pim_dat_img_id_dic,
                    'pim_dat_img_id_inv_dic': pim_dat_img_id_inv_dic,
                    'pim_id_to_aux_feat':pim_id_to_aux_feat,
                    'pim_id_to_title':pim_id_to_title,
                    'pim_id_to_ocr_corrected':pim_id_to_ocr_corrected,
                    'pim_id_to_ner_out': pim_id_to_ner_out,
                    'pim_id_to_brand':pim_id_to_brand,
                    }
        sku_dics = {
                    'sku_id_dic' : sku_id_dic, 
                    'sku_id_inv_dic':sku_id_inv_dic, 
                    'sku_dat_img_id_dic':um_dat_img_id_dic,
                    'sku_dat_img_id_inv_dic': um_dat_img_id_inv_dic,
                    'sku_id_to_aux_feat':sku_id_to_aux_feat,
                    'sku_id_to_title':sku_id_to_title,
                    'sku_id_to_ocr_corrected':sku_id_to_ocr_corrected,
                    'sku_id_to_ner_out': sku_id_to_ner_out,
                    'sku_id_to_brand':sku_id_to_brand,
                    }
        self.sku_dics = sku_dics
        self.pim_dics = pim_dics
        sim_mat_text_ls = [sim_mat_text1, sim_mat_text2, sim_mat_text3, sim_mat_text4]
        sim_mat_img_ls = [sim_mat_img1, sim_mat_img2, sim_mat_img3, sim_mat_img4]
        self.sim_mat_text_ls = sim_mat_text_ls
        self.sim_mat_img_ls = sim_mat_img_ls
        log.info('***** Conducting 1st round of matching *****')

        sku_to_gtin_dic3, quant_feat_mis_matching_skus, partial_mis_matching_skus = match_1_core_multipool(aux_features, sku_id_dic, 
                                                                                                           sim_mat_text_ls, 
                                                                                                           sim_mat_img_ls,
                                                                                                           pim_dics, 
                                                                                                           sku_dics,
                                                                                                           self.params['disabled_features'], 
                                                                                                           max_parallel=1)

        log.info('***** Conducting 2nd round of matching *****')
        
        sku_to_gtin_low_dic3, quant_feat_mis_matching_skus_low, partial_mis_matching_skus_low, no_match_skus = match_2_core_multipool(aux_features, sku_id_dic, 
                                                                                                                                    sim_mat_text_ls, sim_mat_img_ls,
                                                                                                                                       pim_dics, sku_dics, sku_to_gtin_dic3, self.params['disabled_features'], 
                                                                                                                                       max_n= max_n, max_parallel =1)
        log.info('***** Adding OCR based signals  *****')
        self.sku_to_gtin_dic3 = sku_to_gtin_dic3
        self.sku_to_gtin_low_dic3 = sku_to_gtin_low_dic3
        # *****
        #                               build ocr based features
        # *****
        
        corpus2 = build_corpus_for_ocr_feat(df_pim, df_um_tot)
        ocr_based_feat = []
        for i in range(0, len(df_um_tot)):
            feat = get_ocr_based_feat(df_um_tot.loc[i, 'ocr_res_corrected'],df_um_tot.loc[i, 'product_name'], corpus2)
            ocr_based_feat.append(feat)
        df_um_tot['ocr_based_feat'] =  ocr_based_feat

        sku_id_to_ocr_feat = {k:v for k, v in zip(df_um_tot['id'], df_um_tot['ocr_based_feat'])}


        ocr_based_feat = []
        for i in range(0, len(df_pim)):
            feat = get_ocr_based_feat(df_pim.loc[i, 'ocr_res_corrected'],df_pim.loc[i, 'product_name'], corpus2)
            ocr_based_feat.append(feat)
        df_pim['ocr_based_feat'] =  ocr_based_feat
        pim_id_to_ocr_feat = {k:v for k, v in zip(df_pim['id'], df_pim['ocr_based_feat'])}
        # *****
        #                               build kw corpus freq dic
        # *****
        corpus_kw_freq = {}
        for item in df_um_tot['name_cleaned']:
            item = remove_special_characters(item)
            ls=  [x.lower() for x in item.split(' ') if x.isalpha() and len(x)>1]
            for l in ls:
                corpus_kw_freq[l] = corpus_kw_freq.get(l, 0) +1

        for item in df_pim['name_cleaned']:
            item = remove_special_characters(item)
            ls=  [x.lower() for x in item.split(' ') if x.isalpha() and len(x)>1]
            for l in ls:
                corpus_kw_freq[l] = corpus_kw_freq.get(l, 0) +1

        for k,v in corpus_kw_freq.items():
            corpus_kw_freq[k] = v/(len(df_um_tot) +len(df_pim))

        corpus_kw_freq= {k: v for k,v in sorted(corpus_kw_freq.items(), key=lambda item: -1*item[1])}
        # *****
        #                              build freq dic for common kw
        # *****
        common_kw_set= set()
        for k, v in sku_id_to_ocr_feat.items():
            for kw in v['common']:
                common_kw_set.add(kw)

        for k, v in pim_id_to_ocr_feat.items():
            for kw in v['common']:
                common_kw_set.add(kw)


        common_kw_freq_cm = {}

        for k in common_kw_set:
            common_kw_freq_cm[k] = corpus_kw_freq[k]
        common_kw_freq_cm= {k: v for k,v in sorted(common_kw_freq_cm.items(), key=lambda item: -1*item[1])}

        # *************************************************************************************
        #                                   build brand corpus
        # *************************************************************************************

        brand_corpus_dic = {}
        for k, v in sku_id_to_brand.items():
            for vk in v:
                brand_corpus_dic[vk] =brand_corpus_dic.get(vk, 0)+1

        for k, v in pim_id_to_brand.items():
            for vk in v:
                brand_corpus_dic[vk] =brand_corpus_dic.get(vk, 0)+1

        for k,v in brand_corpus_dic.items():
            brand_corpus_dic[k] = v/(len(df_um_tot) +len(df_pim))
        brand_corpus_dic= {k: v for k,v in sorted(brand_corpus_dic.items(), key=lambda item: -1*item[1])}

        brand_corpus = set()
        for k, v in brand_corpus_dic.items():
            if v>=0.01:
                brand_corpus.add(k)

        # *************************************************************************************
        #                                   build unique tokens
        # *************************************************************************************
        unique_tokens = {}
        # kw_to_ignore = ['flavors', 'flavored', 'natural', 'complete', 'picked', 'taste', 'with', 'sweeteners']
        for k,v in common_kw_freq_cm.items():
            if v<0.09:
                if k not in brand_corpus :
                    unique_tokens[k] = v

        # *************************************************************************************
        #                                   Create enhanced matching dictionaries
        # *************************************************************************************

        log.info('***** Enriching the matched dictionaries  *****')

        sku_to_gtin_dic3e = reduce_conf_based_on_ocr(sku_to_gtin_dic3, sku_id_to_title, pim_id_to_title, 
                                                     sku_id_to_ocr_feat, pim_id_to_ocr_feat, 
                                                     sku_id_to_ocr_corrected, pim_id_to_ocr_corrected, unique_tokens)
        sku_to_gtin_low_dic3e = reduce_conf_based_on_ocr(sku_to_gtin_low_dic3, sku_id_to_title, pim_id_to_title, 
                                                     sku_id_to_ocr_feat, pim_id_to_ocr_feat, 
                                                     sku_id_to_ocr_corrected, pim_id_to_ocr_corrected, unique_tokens)

        partial_mis_matching_skus_lowe = reduce_conf_based_on_ocr(partial_mis_matching_skus_low, sku_id_to_title, pim_id_to_title, 
                                                     sku_id_to_ocr_feat, pim_id_to_ocr_feat, 
                                                     sku_id_to_ocr_corrected, pim_id_to_ocr_corrected, unique_tokens)

        partial_mis_matching_skuse = reduce_conf_based_on_ocr(partial_mis_matching_skus, sku_id_to_title, pim_id_to_title, 
                                                     sku_id_to_ocr_feat, pim_id_to_ocr_feat, 
                                                     sku_id_to_ocr_corrected, pim_id_to_ocr_corrected, unique_tokens)

        quant_feat_mis_matching_skuse = reduce_conf_based_on_ocr(quant_feat_mis_matching_skus, sku_id_to_title, pim_id_to_title, 
                                                     sku_id_to_ocr_feat, pim_id_to_ocr_feat, 
                                                     sku_id_to_ocr_corrected, pim_id_to_ocr_corrected, unique_tokens)

        quant_feat_mis_matching_skus_lowe = reduce_conf_based_on_ocr(quant_feat_mis_matching_skus_low, sku_id_to_title, pim_id_to_title, 
                                                     sku_id_to_ocr_feat, pim_id_to_ocr_feat, 
                                                     sku_id_to_ocr_corrected, pim_id_to_ocr_corrected, unique_tokens)
        sku_to_tier = {}
        sku_to_gtin_comb = {}
        for k,v in sku_to_gtin_dic3e.items():
            sku_to_tier[k] ='tier1'
            sku_to_gtin_comb[k] =v
        for k,v in sku_to_gtin_low_dic3e.items():
            sku_to_tier[k] ='tier2'
            assert k not in sku_to_gtin_comb.keys()
            sku_to_gtin_comb[k] =v

        sku_ids = set(df_um_tot['id'])


        no_match_ids = set(no_match_skus.keys())
        high_med_ids= set(sku_to_gtin_comb.keys())
        low_match_ids = set(partial_mis_matching_skus.keys()).union(set(partial_mis_matching_skus_low.keys())) - high_med_ids
        vlow_match_ids = set(quant_feat_mis_matching_skus.keys()).union(set(quant_feat_mis_matching_skus_low.keys()))  - low_match_ids - high_med_ids 
        
        log.info(f"Count of High and medium conf matches: {len(high_med_ids)}")
        log.info(f"Count of Low confidence matches: {len(low_match_ids)}")
        log.info(f"Count of Very Low confidence matches: {len(vlow_match_ids)}")
        log.info(f"Count of No Matches: {len(no_match_ids)}")

        
#         assert high_med_ids.union(low_match_ids.union(vlow_match_ids.union(no_match_ids))) == sku_ids
#         assert high_med_ids.intersection(low_match_ids) ==set()
#         assert high_med_ids.intersection(vlow_match_ids) ==set()
#         assert high_med_ids.intersection(no_match_ids) ==set()
#         assert low_match_ids.intersection(vlow_match_ids) ==set()
#         assert low_match_ids.intersection(no_match_ids) ==set()
#         assert vlow_match_ids.intersection(no_match_ids) ==set()

        # add all inconsistent macthes and theri data
        if high_med_ids.union(low_match_ids.union(vlow_match_ids.union(no_match_ids))) != sku_ids:
            log.info(f"ERROR : Data inconsistent")
        if high_med_ids.intersection(low_match_ids) !=set():
            log.info(f"ERROR : Data inconsistent on low and high intersection, count : {len(high_med_ids.intersection(low_match_ids))}")
        if high_med_ids.intersection(vlow_match_ids) !=set():
            log.info(f"ERROR : Data inconsistent on very low and high intersection, count : {len(high_med_ids.intersection(vlow_match_ids))}")
        if high_med_ids.intersection(no_match_ids) !=set():
            log.info(f"ERROR : Data inconsistent on no match and high intersection, count : {len(high_med_ids.intersection(no_match_ids))}")
        if low_match_ids.intersection(vlow_match_ids) !=set():
            log.info(f"ERROR : Data inconsistent on low and very low intersection, count : {len(low_match_ids.intersection(vlow_match_ids))}")
        if low_match_ids.intersection(no_match_ids) !=set():
            log.info(f"ERROR : Data inconsistent on low and no match intersection, count : {len(low_match_ids.intersection(no_match_ids))}")
        if vlow_match_ids.intersection(no_match_ids) !=set():
            log.info(f"ERROR : Data inconsistent on very low and no match intersection, count : {len(vlow_match_ids.intersection(no_match_ids))}")

        high_med_dic = sku_to_gtin_comb.copy()
        low_dic = {}
        for id in low_match_ids:
            if id in partial_mis_matching_skuse.keys():
                low_dic[id] = partial_mis_matching_skuse[id]
            elif id in partial_mis_matching_skus_lowe.keys():
                low_dic[id] = partial_mis_matching_skus_lowe[id]

        vlow_dic = {}
        for id in vlow_match_ids:
            if id in quant_feat_mis_matching_skuse.keys():
                vlow_dic[id] = quant_feat_mis_matching_skuse[id]
            elif id in quant_feat_mis_matching_skus_lowe.keys():
                vlow_dic[id] = quant_feat_mis_matching_skus_lowe[id]

        no_match_dic = no_match_skus.copy()

        
        log.info('***** Created match dictionaries ordered by Confidence *****')
        
        df_um_tot.loc[df_um_tot['sku_priority']=='Recent & Full Data', 'priority'] = 'p1'
        df_um_tot.loc[df_um_tot['sku_priority']=='Recent & No Image', 'priority'] = 'p2'
        sku_to_pr = {x: y for x, y in zip(df_um_tot['id'], df_um_tot['priority'])}

        case1 =0
        case2 =0
        case3 =0
        case_dic = {}
        for dc in [high_med_dic, low_dic, vlow_dic]:
            for sku_k,v in tqdm(dc.items()):

                sku_img_flag = 0
                if sku_k in sku_id_to_img_path.keys():
                    sku_img_flag = 1
                pim_img_flag =0
                for pim_k, pim_scores in v.items():
                    if pim_k in pim_id_to_img_path.keys():
                        pim_img_flag+=1
                if sku_img_flag==1 and pim_img_flag==len(v.keys()):
                    case_dic[sku_k] = 1
                    case1+=1
                elif sku_img_flag==1 and pim_img_flag<len(v.keys()):
                    case_dic[sku_k] = 2
                    case2+=1
                elif sku_img_flag==0 :
                    case_dic[sku_k] = 3
                    case3+=1


        pim_id_to_gtin = {k: v for k,v in zip(df_pim['id'], df_pim['gtin'])}
        sku_id_to_pid = {k:[v1, v2, v3, v4] for k,v1, v2, v3, v4 in zip(df_um_tot['id'], df_um_tot['product_id'], df_um_tot['retailer'], df_um_tot['url'], df_um_tot['screenshot'])}
        pim_id_to_img_url = {k: v for k,v in zip(df_pim['id'], df_pim['image'])}
        sku_id_to_img_url = {k:v for k,v in zip(df_um_tot['id'], df_um_tot['image'])}
        sku_id_to_pdbid = {k:v1 for k,v1 in zip(df_um_tot['id'], df_um_tot['pdb_id'])}

        high_med_output_dic = {}
        for sku_k in high_med_dic.keys():
            high_med_output_dic[sku_k] = get_conf_decison(high_med_dic[sku_k], dic_type = 'hm')

        low_output_dic = {}
        for sku_k in low_dic.keys():
            low_output_dic[sku_k] = get_conf_decison(low_dic[sku_k], dic_type = 'low')

        vlow_output_dic = {}
        for sku_k in vlow_dic.keys():
            vlow_output_dic[sku_k] = get_conf_decison(vlow_dic[sku_k], dic_type = 'vlow')

        sku_id_to_scrape_gtin = {}
        corrected_gtin_map = prep_corrected_gtin_map(df_pim, 'gtin')
        for i in range(df_um_tot.shape[0]):
            sku_id = df_um_tot.loc[i, 'id']
            candidate = df_um_tot.loc[i, 'gtin']
            sku_id_to_scrape_gtin[sku_id] = search_gtin_in_pim(candidate, corrected_gtin_map)
        
        sku_id_to_matched_gtin = {}
        for i in range(df_um_tot.shape[0]):
            sku_id = df_um_tot.loc[i, 'id']
            candidate = df_um_tot.loc[i, 'matched_gtin']
            sku_id_to_matched_gtin[sku_id] = search_gtin_in_pim(candidate, corrected_gtin_map)

        gtin_to_pim_id = {str(v):k for k,v in pim_id_to_gtin.items()}


        flt_op_cols = ['Timestamp', 'Hex ID', 'Retailer SKU', 'Retailer', 'pdb id', 'SKU Confidence', 
                       'Candidate GTIN', 'Retailer SKU Metadata', 'Candidate GTIN Metadata', 'Preference', 
                       'Match Confidence', 'Already Matched Flag', 'Already Matched GTIN', 'Scrape Based Matched GTIN', 'Brand_change_flag']
        
        flt_dat = []
        log.info('***** Creating Final File  *****')
        sku_id_to_aux_feat = refactor_features(sku_id_to_aux_feat)
        pim_id_to_aux_feat = refactor_features(pim_id_to_aux_feat)
        
        case1 =0
        case2 =0
        case3 =0
        case4 =0

        for dic in [high_med_output_dic, low_output_dic, vlow_output_dic]:
            for sku_k, match_op in dic.items():
                matched_pim_ids = set([match_op['pim_id'][i] for i in range(len(match_op['pim_id']))])
                sku_metadata= {'name': sku_id_to_title[sku_k], 'retailer': sku_id_to_pid[sku_k][1], 
                                'retailer_url': sku_id_to_pid[sku_k][2], 'screenshot':sku_id_to_pid[sku_k][3],
                               'image_url':sku_id_to_img_url[sku_k], 
                               'auxiliary_data': sku_id_to_aux_feat[sku_k]
                               }                       

                already_matched_flag =False
                scrape_matched_gtin ='NULL'
                already_matched_gtin = 'NULL'
                if sku_id_to_scrape_gtin[sku_k] is not None and sku_id_to_matched_gtin[sku_k] is not None:
                    already_matched_flag =True
                    if sku_id_to_scrape_gtin[sku_k] == sku_id_to_matched_gtin[sku_k]:
                        #only one preference 0
                        already_matched_gtin = sku_id_to_matched_gtin[sku_k]
                        scrape_matched_gtin = sku_id_to_scrape_gtin[sku_k]
                        pim_id = gtin_to_pim_id[already_matched_gtin]
                        gtin_metadata= {'name': pim_id_to_title[pim_id], 'image_url':pim_id_to_img_url[pim_id],
                                    'auxiliary_data': pim_id_to_aux_feat[pim_id]
                                   }
                        t = time.localtime()
                        bm_ch_fl =0
                        if pim_id in matched_pim_ids:
                            bm_ch_fl=1
                        current_time = time.strftime("%Y-%m-%d-%H-%M-%S", t)
                        flt_dat.append([current_time, f'{sku_id_to_pid[sku_k][0]}_{sku_id_to_pid[sku_k][1]}_{pim_id_to_gtin[pim_id]}', sku_id_to_pid[sku_k][0], sku_id_to_pid[sku_k][1], sku_id_to_pdbid[sku_k],'high', pim_id_to_gtin[pim_id], sku_metadata, gtin_metadata,0, 'high', already_matched_flag , already_matched_gtin, scrape_matched_gtin, bm_ch_fl])

                        case1+=1
                    else:
                        #two preference 0
                        already_matched_gtin = sku_id_to_matched_gtin[sku_k]
                        scrape_matched_gtin = sku_id_to_scrape_gtin[sku_k]
                        # add already matched gtin entry
                        pim_id = gtin_to_pim_id[already_matched_gtin]
                        gtin_metadata= {'name': pim_id_to_title[pim_id], 'image_url':pim_id_to_img_url[pim_id],
                                    'auxiliary_data': pim_id_to_aux_feat[pim_id]
                                   }
                        bm_ch_fl =0
                        if pim_id in matched_pim_ids:
                            bm_ch_fl=1
                        t = time.localtime()
                        current_time = time.strftime("%Y-%m-%d-%H-%M-%S", t)
                        flt_dat.append([current_time, f'{sku_id_to_pid[sku_k][0]}_{sku_id_to_pid[sku_k][1]}_{pim_id_to_gtin[pim_id]}', sku_id_to_pid[sku_k][0], sku_id_to_pid[sku_k][1], sku_id_to_pdbid[sku_k],'high', pim_id_to_gtin[pim_id], sku_metadata, gtin_metadata,0, 'high', already_matched_flag , already_matched_gtin, scrape_matched_gtin, bm_ch_fl])

                        # add scrape matched gtin entry
                        pim_id = gtin_to_pim_id[scrape_matched_gtin]
                        gtin_metadata= {'name': pim_id_to_title[pim_id], 'image_url':pim_id_to_img_url[pim_id],
                                    'auxiliary_data': pim_id_to_aux_feat[pim_id]
                                   }
                        t = time.localtime()
                        current_time = time.strftime("%Y-%m-%d-%H-%M-%S", t)
                        bm_ch_fl =0
                        if pim_id in matched_pim_ids:
                            bm_ch_fl=1
                        flt_dat.append([current_time, f'{sku_id_to_pid[sku_k][0]}_{sku_id_to_pid[sku_k][1]}_{pim_id_to_gtin[pim_id]}', sku_id_to_pid[sku_k][0], sku_id_to_pid[sku_k][1], sku_id_to_pdbid[sku_k],'high', pim_id_to_gtin[pim_id], sku_metadata, gtin_metadata,0, 'high', already_matched_flag , already_matched_gtin, scrape_matched_gtin, bm_ch_fl])

                        case2+=1


                elif sku_id_to_scrape_gtin[sku_k] is not None and sku_id_to_matched_gtin[sku_k] is None:
                    already_matched_flag =False
                    scrape_matched_gtin =sku_id_to_scrape_gtin[sku_k]
                    pim_id = gtin_to_pim_id[scrape_matched_gtin]
                    gtin_metadata= {'name': pim_id_to_title[pim_id], 'image_url':pim_id_to_img_url[pim_id],
                                'auxiliary_data': pim_id_to_aux_feat[pim_id]
                               }
                    bm_ch_fl =0
                    if pim_id in matched_pim_ids:
                        bm_ch_fl=1
                    t = time.localtime()
                    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", t)
                    flt_dat.append([current_time, f'{sku_id_to_pid[sku_k][0]}_{sku_id_to_pid[sku_k][1]}_{pim_id_to_gtin[pim_id]}', sku_id_to_pid[sku_k][0], sku_id_to_pid[sku_k][1], sku_id_to_pdbid[sku_k],match_op['overall_conf'], pim_id_to_gtin[pim_id], sku_metadata, gtin_metadata,0, match_op['overall_conf'], already_matched_flag ,already_matched_gtin, scrape_matched_gtin, bm_ch_fl])

                    case3+=1
                elif sku_id_to_scrape_gtin[sku_k] is None and sku_id_to_matched_gtin[sku_k] is not None:
                    already_matched_flag = True
                    already_matched_gtin =sku_id_to_matched_gtin[sku_k]
                    pim_id = gtin_to_pim_id[already_matched_gtin]
                    gtin_metadata= {'name': pim_id_to_title[pim_id], 'image_url':pim_id_to_img_url[pim_id],
                                'auxiliary_data': pim_id_to_aux_feat[pim_id]
                               }
                    t = time.localtime()
                    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", t)
                    bm_ch_fl =0
                    if pim_id in matched_pim_ids:
                        bm_ch_fl=1
                    flt_dat.append([current_time, f'{sku_id_to_pid[sku_k][0]}_{sku_id_to_pid[sku_k][1]}_{pim_id_to_gtin[pim_id]}', sku_id_to_pid[sku_k][0], sku_id_to_pid[sku_k][1], sku_id_to_pdbid[sku_k],'high', pim_id_to_gtin[pim_id], sku_metadata, gtin_metadata,0, 'high', already_matched_flag , already_matched_gtin, scrape_matched_gtin, bm_ch_fl])

                    case4+=1

                cnt =1
                for i in range(len(match_op['pim_id'])):

                    pim_id = match_op['pim_id'][i]
                    if (scrape_matched_gtin!='NULL' and gtin_to_pim_id[scrape_matched_gtin] == pim_id) or (already_matched_gtin!='NULL' and gtin_to_pim_id[already_matched_gtin] == pim_id):
                        continue

                    gtin_metadata= {'name': pim_id_to_title[pim_id], 'image_url':pim_id_to_img_url[pim_id],
                                'auxiliary_data': pim_id_to_aux_feat[pim_id]
                               }
                    t = time.localtime()
                    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", t)
                    overall_conf = match_op['overall_conf']
                    if already_matched_gtin!='NULL':
                        overall_conf= 'high'
                    flt_dat.append([current_time, f'{sku_id_to_pid[sku_k][0]}_{sku_id_to_pid[sku_k][1]}_{pim_id_to_gtin[pim_id]}', sku_id_to_pid[sku_k][0], sku_id_to_pid[sku_k][1], sku_id_to_pdbid[sku_k],overall_conf, pim_id_to_gtin[pim_id], sku_metadata, gtin_metadata, cnt, match_op['candidate_conf'][i], already_matched_flag , already_matched_gtin, scrape_matched_gtin, 1])
                    cnt+=1
        self.sku_id_to_pid = sku_id_to_pid
        self.pim_id_to_gtin = pim_id_to_gtin
        log.info(f"Case 1: (scrape based gtin = matched gtin) = {case1}")
        log.info(f"Case 2: (scrape based gtin != matched gtin) = {case2}")
        log.info(f"Case 3: (only scrape based gtin present) = {case3}")
        log.info(f"Case 4: (only matched based gtin present) = {case4}")
        # log.info(len(flt_dat))
        df_flt = pd.DataFrame(flt_dat, columns = flt_op_cols )
        df_flt.loc[df_flt['Match Confidence']=='high', 'Match Confidence'] = 'High'
        df_flt.loc[df_flt['Match Confidence']=='medium', 'Match Confidence'] = 'Medium'
        df_flt.loc[df_flt['Match Confidence']=='low', 'Match Confidence'] = 'Low'
        df_flt.loc[df_flt['Match Confidence']=='very low', 'Match Confidence'] = 'Very Low'
        df_flt.loc[df_flt['SKU Confidence']=='high', 'SKU Confidence'] = 'High'
        df_flt.loc[df_flt['SKU Confidence']=='medium', 'SKU Confidence'] = 'Medium'
        df_flt.loc[df_flt['SKU Confidence']=='low', 'SKU Confidence'] = 'Low'
        df_flt.loc[df_flt['SKU Confidence']=='very low', 'SKU Confidence'] = 'Very Low'
        self.df_flt_o = df_flt
        df_flt = self.beautify_brand(df_flt)
        df_flt = df_flt.drop(columns = ['Brand_change_flag'])
        flat_file_path = os.path.join(self.params['dirs']['processed_files_folder'], f'{client_name}_matching_flat_file.json')
        df_flt['pdb id'] = df_flt['pdb id'].apply(lambda x : str(x))
        df_flt['Candidate GTIN']= df_flt['Candidate GTIN'].apply(lambda x : str(x))
        df_flt['Already Matched GTIN'] = df_flt['Already Matched GTIN'].apply(lambda x : str(x))
        df_flt['Scrape Based Matched GTIN'] = df_flt['Scrape Based Matched GTIN'].apply(lambda x : str(x))
        df_flt.to_json(flat_file_path, orient="records")
        s3_upload_data(flat_file_path, client_name, run_date, processed=True)
        log.info('***** Output File uploaded to S3 *****')
        feat_set = set()
        pref_dic = {}
        mand_dic = {}
        pref =1
        for k in self.params['mandatory_features']:
            if k not in feat_set:
                mand_dic[k] = pref
                pref+=1
                feat_set.add(k)
        pref_dic['mandatory features'] = mand_dic
        nmand_dic = {}
        pref = 1
        for k in self.params['llama_features'] + self.params['kw_features']:
            if k not in feat_set:
                nmand_dic[k] = pref
                pref+=1
                feat_set.add(k)
        pref_dic['non mandatory features'] = nmand_dic
        pref_file_path = os.path.join(self.params['dirs']['processed_files_folder'], f'{client_name}_feature_priority_map.json')
        with open(pref_file_path, "w") as outfile: 
            json.dump(pref_dic, outfile)
        s3_upload_data(pref_file_path, client_name, run_date, processed=True)
        log.info('***** Feature Preference File uploaded to S3 *****')
#         Upload features file and matching flat file to GCP
        upload_matching_file(flat_file_path, client_name, run_date)
        upload_feature_file(pref_file_path, client_name)
        log.info('Output files uploaded to GCP')
        self.df_flt = df_flt

import os
import numpy as np
import pandas as pd
import mapply
from utils.DataFetchHelper import *
from utils.TextProcessing import *
from utils.ImageFuncs import *
from constants.FileNameConstants import *
from constants.S3Constants import *
from utils.UniNERHelper import *
from utils.FlanT5Helper import *
from utils.OCRImpHelper import *
from utils.PaddleOCRExtraction import *
from utils.LlamaHelper import *
from utils.DataEnrichmentHelper import *
from utils.GeneralHelper import *
from utils.MatchingAuxHelper import *
from utils.ConfigHelper import *
from utils.log import Log    
from utils.Slack import *
from utils.SlackDecorator import *
from utils.TranslationHelper import *
from constants.SlackConstants import SlackConstants
from utils.GModel import *
from utils.decorators import error_decorator
Slack_=Slack(SlackConstants.bot_token)
log = Log(__name__)

# from utils.Initialize import Initialize
# Initialize().download_config()

class DataProcessing:
    #@error_decorator
    def __init__(self, client_name, run_date, override):
        if override==None:
            override= 'false'
        if override.lower() =='true' or override.lower() == 'y' or override.lower() =='yes':
            self.override = True
        else:
            self.override = False
        self.client_name = client_name
        self.run_date = run_date
        self.ensure_folders()
        self.prepare_old_cache_file_paths()
        config_helper = ConfigHelper(self.client_name)
        created_params = config_helper.create_feature_params()
        for k,v in created_params.items():
            self.params[k] = v
        try:
            df_total = load_unmatched_data(self.client_name)
            self.df_total = df_total
            self.df_total = self.df_total.dropna(subset = ['product_name']).reset_index(drop=True)
            #if len(self.df_total)==0:
            #    raise Exception('SKU Data Empty')
            #log.info(f"df_tot_columns {self.df_total.columns}")
        except Exception as e:
            log.info("***** Error in querying total unmatched data *****")
            log.info(e)

        try:
            try:
                self.df_pim = load_pim_data(self.client_name)
            except Exception as e:
                log.info('could not query pim data')
                log.info(e)
            
            self.df_pim = self.df_pim.sort_values(by = ['status', 'image'])
            self.df_pim = self.df_pim.drop_duplicates(subset = ['pim_config', 'gtin', 'name']).reset_index(drop = True)
            if 'retailer' in self.df_pim.columns:
                self.df_pim['retailer'] = self.df_pim['retailer'].fillna('NONE')
            else:
                self.df_pim['retailer'] = 'NONE'
            lang_map_df_path = os.path.join(CONSTANTS,RETAILER_LANG_CODE_MAP)
            lang_map_df = pd.read_csv(lang_map_df_path)
            lang_map = {k:v for k,v in zip(lang_map_df['retailer'], lang_map_df['retailer language code'])}
            self.df_pim['lang'] = self.df_pim['retailer'].apply(lambda x: lang_map.get(x, None))
            self.df_pim = self.df_pim.sort_values(by = ['status', 'image', 'lang'])
            self.df_pim = self.df_pim.drop_duplicates(subset = ['pim_config','gtin']).reset_index(drop = True)
            self.df_pim = self.df_pim.rename(columns = {'name':'product_name'})
            self.df_pim['id'] = ['pim_id_'+str(x) for x in self.df_pim.index]
            self.df_pim = self.df_pim.dropna(subset = ['product_name']).reset_index(drop=True)
            #log.info(f"df_pim columns {self.df_pim.columns}")
        except Exception as e:
            log.info("***** Error in loading PIM data *****")
            log.info(e)
            
        ## translate if data other than english
        try:
            #check en lang in pim and overwrite
            if 'en' in set(self.df_pim['lang'].unique()):
                temp = self.df_pim.loc[self.df_pim['lang']=='en', :].sample(min(20, len(self.df_pim.loc[self.df_pim['lang']=='en'])))
                sample_titles = temp['product_name'].to_list()
                gModel = GModel()
                output = []
                for i in tqdm(sample_titles):
                    op = gModel.translate_util(text = i, src_lang = 'UNKNOWN')
                    output.append(op)
                english_flag = True
                for op in output:
                    if op['detectedSourceLanguage'] !='en':
                        english_flag= False
                if english_flag!=True:
                    self.df_pim.loc[self.df_pim['lang']=='en', 'lang'] = None
            #check en lang in unmatched and overwrite
            if 'en' in set(self.df_total['lang'].unique()):
                temp = self.df_total.loc[self.df_total['lang']=='en', :].sample(min(20, len(self.df_total.loc[self.df_total['lang']=='en'])))
                sample_titles = temp['product_name'].to_list()
                gModel = GModel()
                output = []
                for i in tqdm(sample_titles):
                    op = gModel.translate_util(text = i, src_lang = 'UNKNOWN')
                    output.append(op)
                english_flag = True
                for op in output:
                    if op['detectedSourceLanguage'] !='en':
                        english_flag= False
                if english_flag!=True:
                    self.df_total.loc[self.df_total['lang']=='en', 'lang'] = None
                
            lang_present = set(self.df_pim['lang'].unique()).union(set(self.df_total['lang'].unique())) - set(['en'])
            
            if len(lang_present)>0:
                df_pim_cp = self.df_pim.copy()
                df_sku_cp = self.df_total.copy()
                
                log.info(lang_present)
                gcp_cache_read_success = False
                try:
                    
                    title_translation_output_not_exists_sku, title_translation_output_exists_sku = self.absorb_gcp_cache()
                    title_translation_output_exists_sku =title_translation_output_exists_sku.rename(columns = {'id':'tr_id', 'title':'product_name', 
                                                      'translatedText':'translated_title_extracted', 
                                                     'detectedSourceLanguage':'lang'})
                    title_translation_output_exists_sku['dat_type'] = 'sku'
                    title_translation_output_exists_sku['lang2'] = 'en'
                    title_translation_output_exists_sku['translated_title'] = title_translation_output_exists_sku['translated_title_extracted'].apply(lambda x: {'detectedSourceLanguage':'unknown','translatedText':x })
                    title_translation_output_exists_sku= title_translation_output_exists_sku[['tr_id', 'product_name', 'lang', 'dat_type', 'model_type','translated_title', 'translated_title_extracted', 'lang2']]
                    gcp_cache_read_success=True
                except:
                    log.info("********* Error in reading cache from GCP ***************")
                    pass
                if gcp_cache_read_success==False:
                    trunc_df, object_name = prepare_text_data_and_upload_to_s3(self.client_name, self.df_total.copy(), self.df_pim.copy(), self.run_date )
                else:
                    df_sku_cp['tr_id'] = df_sku_cp['retailer'].astype(str) + "-" +df_sku_cp['product_id'].astype(str)
                    df_sku_cp_tr = df_sku_cp.loc[~df_sku_cp['tr_id'].isin(title_translation_output_exists_sku['tr_id'].to_list()), :].reset_index(drop=True)
                    df_sku_cp_tr= df_sku_cp_tr.drop(columns = ['tr_id'])
                    trunc_df, object_name = prepare_text_data_and_upload_to_s3(self.client_name, df_sku_cp_tr.copy(), self.df_pim.copy(), self.run_date )

                cache_file_path = os.path.join(PROCESSED_FOLDER_S3, self.client_name, CACHED_FOLDER, f'{self.client_name}_translation_res.parquet')
                op = generate_translation_results(self.client_name , 'google', 
                                                 object_name, 
                                                 cache_s3_loc = cache_file_path, 
                                                 lang = 'mix',
                                                 bucket = BUCKET)
                check_translation_status(op)
                s3_translation_output_path = object_name.split('.')
                s3_translation_output_path = s3_translation_output_path[0] + '_output' + '.parquet'
                local_translation_output_path = os.path.join(self.params['dirs']['cached_files_folder'], 
                                                       os.path.split(s3_translation_output_path)[1])
                s3_download_file(s3_translation_output_path, local_translation_output_path)
                tr_op = pd.read_parquet(local_translation_output_path)
                df_pim_tr_op = tr_op.loc[tr_op['dat_type']=='pim'].reset_index(drop=True)
                df_sku_tr_op = tr_op.loc[tr_op['dat_type']=='sku'].reset_index(drop=True)
                df_sku_tr_op = df_sku_tr_op.rename(columns = {'id': 'tr_id', 'title':'product_name'})
                df_pim_tr_op = df_pim_tr_op.rename(columns = {'id': 'tr_id', 'title':'product_name'})
                df_sku_tr_op['translated_title_extracted'] = df_sku_tr_op['translated_title'].apply(lambda x: x['translatedText'])
                df_pim_tr_op['translated_title_extracted'] = df_pim_tr_op['translated_title'].apply(lambda x: x['translatedText'])
                df_sku_tr_op['lang2'] = df_sku_tr_op['translated_title'].apply(lambda x: x['detectedSourceLanguage'])
                df_pim_tr_op['lang2'] = df_pim_tr_op['translated_title'].apply(lambda x: x['detectedSourceLanguage'])
                
                if gcp_cache_read_success:
                    df_sku_tr_op= pd.concat([df_sku_tr_op, title_translation_output_exists_sku], ignore_index=True)
                    
                    
                df_sku_cp['retailer'].fillna(client_name, inplace= True)
                df_sku_cp['tr_id'] = df_sku_cp['retailer'].astype(str) + "-" +df_sku_cp['product_id'].astype(str)
                df_pim_cp['pim_config'].fillna(client_name, inplace = True)
                df_pim_cp['tr_id'] = df_pim_cp['pim_config'].astype(str) + "-" + df_pim_cp['gtin'].astype(str)
                
                df_sku_cp= pd.merge(df_sku_cp,df_sku_tr_op[['tr_id', 'product_name', 'translated_title_extracted', 'lang2']], 
                                    on = ['tr_id', 'product_name'], 
                                    how = 'left')
                df_sku_cp.loc[df_sku_cp['translated_title_extracted'].isna(), 'translated_title_extracted'] = df_sku_cp.loc[df_sku_cp['translated_title_extracted'].isna(), 'product_name']
                
                df_pim_cp= pd.merge(df_pim_cp,df_pim_tr_op[['tr_id', 'product_name', 'translated_title_extracted', 'lang2']], 
                                    on = ['tr_id', 'product_name'], 
                                    how = 'left')
                df_pim_cp.loc[df_pim_cp['translated_title_extracted'].isna(), 'translated_title_extracted'] = df_pim_cp.loc[df_pim_cp['translated_title_extracted'].isna(), 'product_name']
                
                df_pim_cp.loc[df_pim_cp['lang'].isna(), 'lang'] = df_pim_cp.loc[df_pim_cp['lang'].isna(), 'lang2']
                df_sku_cp.loc[df_sku_cp['lang'].isna(), 'lang'] = df_sku_cp.loc[df_sku_cp['lang'].isna(), 'lang2']
                
                df_pim_cp = df_pim_cp.rename(columns = {'product_name':'old_product_name', 'translated_title_extracted': 'product_name'})
                df_sku_cp = df_sku_cp.rename(columns = {'product_name':'old_product_name', 'translated_title_extracted': 'product_name'})
                df_pim_cp = df_pim_cp.drop(columns = ['lang2', 'tr_id'])
                df_sku_cp = df_sku_cp.drop(columns = ['lang2', 'tr_id'])
                log.info(f'columns added in PIM Data {set(df_pim_cp.columns) - set(self.df_pim.columns)}')
                log.info(f'columns added in SKU Data {set(df_sku_cp.columns) - set(self.df_total.columns)}')
                self.df_pim = df_pim_cp
                self.df_total = df_sku_cp

        except Exception as e:
            log.info("***** Error in translation *****")
            log.info(e)
            raise Exception('***** Error in Translation ***** '+ str(e))
            
    def absorb_gcp_cache(self):
        title_translation_df = self.df_total.copy()
        title_translation_df['id'] = title_translation_df['retailer']+'_'+title_translation_df['product_id']
        # Product ID/Retailer Mapping
        id_prodid_retailer_map = {k:[v1, v2] for k,v1,v2 in zip(title_translation_df['id'], title_translation_df['retailer'], 
                                                                title_translation_df['product_id'])}     
        title_translation_df = title_translation_df.drop(columns = ['retailer', 'product_id'])
        title_translation_df = title_translation_df[['id', 'product_name']]
        title_translation_df.columns = ['id', 'title']
        title_translation_df['model_type'] = 'google' 

        # add gcp cache
        gcp_cache = get_gcp_cache()
        #gcp_cache = pd.read_parquet('gcp_cache.parquet')
        log.info("GCP cache read complete")
        gcp_cache = gcp_cache.rename(columns = {'NAME':'title'})
        gcp_cache = gcp_cache.rename(columns = {'TRANSLATED_NAME':'translatedText'})
        gcp_cache['model_type'] = 'google'
        gcp_cache['id'] = gcp_cache['RETAILER'] + '_' + gcp_cache['PRODUCT_ID']
        gcp_cache = gcp_cache.drop(columns = ['RETAILER', 'PRODUCT_ID'])
        gcp_cache = gcp_cache[['id', 'title', 'model_type', 'translatedText']]
        gcp_cache['detectedSourceLanguage'] = 'UNKNOWN'

         # Checking cache on big query output
        log.info("Checking cache with Big Query cache")
        title_translation_output_not_exists_sku, title_translation_output_exists_sku = check_title_translation_cache_on_bigquery(title_translation_df, gcp_cache)
        return title_translation_output_not_exists_sku, title_translation_output_exists_sku

    def prepare_old_cache_file_paths(self):
        bucket_name = BUCKET
        cache_file_patterns = {
            'sku_p1': r'/pipeline-runs/'+f'{self.client_name}'+r'/cached_data/\d{4}/\d{2}/\d{2}/'+f'{self.client_name}'+r'_post_ner_mismatches\.parquet',
            'sku_p2': r'/pipeline-runs/'+f'{self.client_name}'+r'/cached_data/\d{4}/\d{2}/\d{2}/'+f'{self.client_name}'+r'_post_images_post_ner_mismatches\.parquet',
            'sku_p3': r'/pipeline-runs/'+f'{self.client_name}'+r'/cached_data/\d{4}/\d{2}/\d{2}/'+f'{self.client_name}'+r'_mismatches_post_uni_ner\.parquet',
            'sku_p4': r'/pipeline-runs/'+f'{self.client_name}'+r'/cached_data/\d{4}/\d{2}/\d{2}/'+f'{self.client_name}'+r'_mismatches_post_flan\.parquet',
            'sku_p5': r'/pipeline-runs/'+f'{self.client_name}'+r'/cached_data/\d{4}/\d{2}/\d{2}/'+f'{self.client_name}'+r'_mismatches_post_llama\.parquet',
            'sku_p6': r'/pipeline-runs/'+f'{self.client_name}'+r'/cached_data/\d{4}/\d{2}/\d{2}/'+f'{self.client_name}'+r'_mismatches_post_ocr\.parquet',
            'sku_p7': r'/pipeline-runs/'+f'{self.client_name}'+r'/processed_data/\d{4}/\d{2}/\d{2}/'+f'{self.client_name}'+r'_processed_mismatches\.parquet',
            'pim_p1': r'/pipeline-runs/'+f'{self.client_name}'+r'/cached_data/\d{4}/\d{2}/\d{2}/'+f'{self.client_name}'+r'_post_ner_pim\.parquet',
            'pim_p2': r'/pipeline-runs/'+f'{self.client_name}'+r'/cached_data/\d{4}/\d{2}/\d{2}/'+f'{self.client_name}'+r'_post_images_post_ner_pim\.parquet',
            'pim_p3': r'/pipeline-runs/'+f'{self.client_name}'+r'/cached_data/\d{4}/\d{2}/\d{2}/'+f'{self.client_name}'+r'_pim_post_uni_ner\.parquet',
            'pim_p4': r'/pipeline-runs/'+f'{self.client_name}'+r'/cached_data/\d{4}/\d{2}/\d{2}/'+f'{self.client_name}'+r'_pim_post_flan\.parquet',
            'pim_p5': r'/pipeline-runs/'+f'{self.client_name}'+r'/cached_data/\d{4}/\d{2}/\d{2}/'+f'{self.client_name}'+r'_pim_post_llama\.parquet',
            'pim_p6': r'/pipeline-runs/'+f'{self.client_name}'+r'/cached_data/\d{4}/\d{2}/\d{2}/'+f'{self.client_name}'+r'_pim_post_ocr\.parquet',
            'pim_p7': r'/pipeline-runs/'+f'{self.client_name}'+r'/processed_data/\d{4}/\d{2}/\d{2}/'+f'{self.client_name}'+r'_processed_pim\.parquet',
        }
        prefix = f'pim-matching/pipeline-runs/{self.client_name}/'
        log.info("************************ Listing all S3 files for older runs **********************")
        file_paths = list_s3_files(bucket_name, prefix)
        log.info("************************ Creating Cache file path dict  ***************************")
        cache_file_path_dic = {}
        for file_type, pattern in cache_file_patterns.items():
            filtered_paths = [path for path in file_paths if re.search(pattern, path)]
            cache_file_path_dic[file_type] = get_latest_file_path(filtered_paths, self.run_date)
        self.cache_file_path_dic = cache_file_path_dic
        log.info(f"cache file path dict :{cache_file_path_dic}")
        

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
    
    @error_decorator
    @slack_decorator 
    def run(self):
        override = self.override 
        #ocrobj = OCR()
        client_name = self.client_name
        run_date = self.run_date
        image_download_occured_l1 = False
        image_download_occured_l2 = False
        self.df_total=self.df_total.loc[:, ~self.df_total.columns.str.contains('^Unnamed')]
        self.df_pim=self.df_pim.loc[:, ~self.df_pim.columns.str.contains('^Unnamed')]
        self.df_total= self.df_total.dropna(subset = ['product_name']).reset_index(drop=True)
        self.df_pim= self.df_pim.dropna(subset = ['product_name']).reset_index(drop=True)
        if len(self.df_pim)==0:
            raise Exception('PIM Data Empty')
        if len(self.df_total)==0:
            raise Exception('SKU Data Empty')
        #================================ Processs Unmatched data ========================================   
        column_set = set()
        new_columns = set()
        try:
            unmatched_total_c1 = os.path.join(self.params['dirs']['cached_files_folder'], f'{self.client_name}_post_ner_mismatches.parquet')
            s3_path = get_s3_path(unmatched_total_c1, client_name, run_date, processed=False)
            if os.path.exists(unmatched_total_c1)==False and s3_object_exists(s3_path):
                s3_download_file(s3_path, unmatched_total_c1)
            if os.path.exists(unmatched_total_c1) and override==False:
                df_total2 = pd.read_parquet(unmatched_total_c1)
                #s3_upload_data(unmatched_total_c1, client_name, run_date, processed=False)
            else:
                self.df_total = self.df_total.reset_index(drop=True)
                log.info('***** Cleaning Titles for unmatched data *****')   
                df_total1 = self.df_total.copy()
                if self.cache_file_path_dic['sku_p1'] is not None:
                    log.info("************* Adding data from cache *************")
                    local = os.path.join(self.params['dirs']['cached_files_folder'], f'{self.client_name}_post_ner_mismatches_old.parquet')
                    s3_download_file(self.cache_file_path_dic['sku_p1'][1], local)
                    df_old1 = pd.read_parquet(local)
                    df_old1 = df_old1.loc[:, ~df_old1.columns.str.contains('^Unnamed')]
                    relevant_cols = ['pdb_id', 'name_cleaned', 'ner_out', 'ner_out_updated', 'size_regex']
                    df_old1 = df_old1.dropna(subset = ['name_cleaned', 'pdb_id']).reset_index(drop = True)
                    df_req = df_total1.loc[~df_total1['pdb_id'].isin(df_old1['pdb_id'].to_list()), :].reset_index(drop=True)
                    df_cache = df_total1.loc[df_total1['pdb_id'].isin(df_old1['pdb_id'].to_list()), :].reset_index(drop=True)
                    log.info(f"columns to be added from cache: {set(df_old1.columns) - set(df_cache.columns)}")
                    df_cache = pd.merge(df_cache, df_old1.loc[:,relevant_cols], on = 'pdb_id', how = 'left')
                    list_cols = ['ner_out', 'ner_out_updated']
                    log.info(f"cache found for {len(df_cache)} entries, computing for {len(df_req)} entries")
                    for col in list_cols:
                        df_cache[col] = df_cache[col].apply(lambda x: x.tolist())
                    if len(df_req)>0:
                        df_req_op = get_cleaned_title(df_req.copy(), client_name)
                        df_req_op['size_regex'] = df_req_op['ner_out_updated'].map(get_size_regex)
                        for i in range(len(df_req_op)):
                            p = re.search(r'(\d+(?:\.\d+)?)\s*-?\s*(fz)', df_req_op.loc[i, 'product_name'], re.IGNORECASE)
                            if p!=None:
                                df_req_op.loc[i, 'product_name'] = re.sub(r'(\d+(?:\.\d+)?)\s*-?\s*(fz)', f'{p.group(1)} oz', df_req_op.loc[i, 'product_name'], flags=re.I)
                        df_total2 = pd.concat([df_cache, df_req_op], ignore_index=True)
                        df_total2 = df_total2.sort_values(by =['id']).reset_index(drop=True)
                    else:
                        df_total2 = df_cache.sort_values(by =['id']).reset_index(drop=True)
                else:
                    #log.info('***** Cleaning Titles for unmatched data *****')
                    df_total2 = get_cleaned_title(df_total1.copy(), client_name)
                    df_total2['size_regex'] = df_total2['ner_out_updated'].map(get_size_regex)
                    for i in range(len(df_total2)):
                        p = re.search(r'(\d+(?:\.\d+)?)\s*-?\s*(fz)', df_total2.loc[i, 'product_name'], re.IGNORECASE)
                        if p!=None:
                            df_total2.loc[i, 'product_name'] = re.sub(r'(\d+(?:\.\d+)?)\s*-?\s*(fz)', f'{p.group(1)} oz', df_total2.loc[i, 'product_name'], flags=re.I)
                    df_total2 = df_total2.sort_values(by =['id']).reset_index(drop=True)
                    
                df_total2.to_parquet(unmatched_total_c1, index = False)
                s3_upload_data(unmatched_total_c1, client_name, run_date, processed=False)
            
        except Exception as e:
            log.info('***** Error in Cleaning Titles for unmatched data *****')
            log.info(e)
            raise Exception('***** Error in Cleaning Titles for unmatched data ***** '+ str(e))
            
        try:
            unmatched_total_c2 = os.path.join(self.params['dirs']['cached_files_folder'], f'{self.client_name}_post_images_post_ner_mismatches.parquet')
            local_path = unmatched_total_c2
            s3_path = get_s3_path(local_path, client_name, run_date, processed=False)
            if os.path.exists(local_path)==False and s3_object_exists(s3_path):
                s3_download_file(s3_path, local_path)
            if os.path.exists(unmatched_total_c2) and override==False:
                df_total3 = pd.read_parquet(unmatched_total_c2)
                #s3_upload_data(unmatched_total_c2, client_name, run_date, processed=False)
            else:

                image_download_occured_l1 = True
                log.info('***** Downloading Images for unmatched data *****')
                df_total3 = download_images_create_flag(df_total2)
                df_total3 = remove_white_space_all(df_total3)
                log.info(f"Image download failed for  {df_total3.loc[df_total3['fail']==1, 'image_path'].isna().sum()}")
                log.info(f"Fail distribution {df_total3['fail'].value_counts().to_dict()}")
                df_total3.to_parquet(unmatched_total_c2, index = False)
                s3_upload_data(unmatched_total_c2, client_name, run_date, processed=False)
            
        except Exception as e:
            image_download_occured_l1 = False
            log.info('***** Error in Downloading Images for unmatched data *****')
            log.info(e)
            raise Exception('***** Error in Downloading Images for unmatched data ***** '+ str(e))
                      

        try:
        ### create image embeddings for unmatched data
            local_path = os.path.join(self.params['dirs']['processed_files_folder'], f'{self.client_name}_sku_raw_emb.parquet')
            s3_path = get_s3_path(local_path, client_name, run_date, processed=True)
            if s3_object_exists(s3_path) == False:
                if os.path.exists(os.path.join(self.params['dirs']['processed_files_folder'], f'{self.client_name}_sku_raw_emb.parquet'))== False or override:
                    if image_download_occured_l1 == False:
                        log.info('***** Downloading Images for unmatched data *****')
                        temp = download_images_create_flag(df_total2)
                        temp = remove_white_space_all(temp)
                        image_download_occured_l1 = True
                    create_and_save_image_embeddings(df_total3, self.params['dirs']['processed_files_folder'], self.client_name, file_type = 'sku', emb_type = 'raw')
                if s3_upload_data(os.path.join(self.params['dirs']['processed_files_folder'], f'{self.client_name}_sku_raw_emb.parquet'), self.client_name, self.run_date, processed=True)==1:
                    log.info("Raw SKU embeddings uploaded successfully")

            local_path = os.path.join(self.params['dirs']['processed_files_folder'], f'{self.client_name}_sku_pro_emb.parquet')
            s3_path = get_s3_path(local_path, client_name, run_date, processed=True)
            if s3_object_exists(s3_path) == False:
                if os.path.exists(os.path.join(self.params['dirs']['processed_files_folder'],f'{self.client_name}_sku_pro_emb.parquet'))== False or override:
                    if image_download_occured_l1 == False:
                        log.info('***** Downloading Images for unmatched data *****')
                        temp = download_images_create_flag(df_total2)
                        temp = remove_white_space_all(temp)
                        image_download_occured_l1 = True
                    create_and_save_image_embeddings(df_total3, self.params['dirs']['processed_files_folder'], self.client_name, file_type = 'sku', emb_type = 'pro')
                if s3_upload_data(os.path.join(self.params['dirs']['processed_files_folder'], f'{self.client_name}_sku_pro_emb.parquet'), self.client_name, self.run_date, processed=True) ==1:
                    log.info("Pro SKU embeddings uploaded successfully")
        except Exception as e:
            log.info('***** Error in Creating SKU Image Embeddings *****')
            log.info(e)   
            raise Exception('***** Error in Creating SKU Image Embeddings ***** '+ str(e))

        df_total4 = df_total3
        # try:
#             unmatched_total_c3 = os.path.join(self.params['dirs']['cached_files_folder'], f'{self.client_name}_mismatches_post_uni_ner.parquet')
#             local_path = unmatched_total_c3
#             s3_path = get_s3_path(local_path, client_name, run_date, processed=False)
#             if os.path.exists(local_path)==False and s3_object_exists(s3_path):
#                 s3_download_file(s3_path, local_path)
#             if os.path.exists(unmatched_total_c3) and override==False:
#                 df_total4 = pd.read_parquet(unmatched_total_c3)
#                 #s3_upload_data(unmatched_total_c3, client_name, run_date, processed=False)
                
#             else:
#                 log.info('***** Getting NER for unmatched data*****')
#                 if self.cache_file_path_dic['sku_p3'] is not None:
#                     log.info("************* Adding data from cache *************")
#                     local = os.path.join(self.params['dirs']['cached_files_folder'], f'{self.client_name}_mismatches_post_uni_ner_old.parquet')
#                     s3_download_file(self.cache_file_path_dic['sku_p3'][1], local)
#                     df_old1 = pd.read_parquet(local)
#                     df_old1 = df_old1.loc[:, ~df_old1.columns.str.contains('^Unnamed')]
#                     relevant_cols = ['pdb_id', 'uni_ner_brand']
#                     df_old1 = df_old1.dropna(subset = ['pdb_id']).reset_index(drop = True)
#                     df_req = df_total3.loc[~df_total3['pdb_id'].isin(df_old1['pdb_id'].to_list()), :].reset_index(drop=True)
#                     df_cache = df_total3.loc[df_total3['pdb_id'].isin(df_old1['pdb_id'].to_list()), :].reset_index(drop=True)
#                     log.info(f"columns to be added from cache: {set(df_old1.columns) - set(df_cache.columns)}")
#                     df_cache = pd.merge(df_cache, df_old1.loc[:,relevant_cols], on = 'pdb_id', how = 'left')
#                     list_cols = ['uni_ner_brand']
#                     log.info(f"cache found for {len(df_cache)} entries, computing for {len(df_req)} entries")
#                     for col in list_cols:
#                         df_cache[col] = df_cache[col].apply(lambda x: x.tolist())
#                     if len(df_req)>0:
#                         uniner = UniNER(self.client_name)
#                         df_req_op = uniner.get_brand_multipool(df_req.copy())
#                         df_total4 = pd.concat([df_cache, df_req_op], ignore_index=True)
#                         df_total4 = df_total4.sort_values(by =['id']).reset_index(drop=True)
#                     else:
#                         df_total4 = df_cache.sort_values(by =['id']).reset_index(drop=True)
#                 else:
#                     uniner = UniNER(self.client_name)
#                     df_total4 = df_total3.copy()
#                     #for feature in self.params['uni_ner_features']:
#                         #df_total4 = uniner.get_feature_multipool(df_total4.copy(), feature) 
#                     df_total4 = uniner.get_brand_multipool(df_total4.copy())
#                 df_total4.to_parquet(unmatched_total_c3, index = False)
#                 s3_upload_data(unmatched_total_c3, client_name, run_date, processed=False)
                
#         except Exception as e:
#             log.info('***** Error in Getting NER for unmatched data *****')
#             log.info(e)
#             raise Exception('***** Error in Getting NER for unmatched data ***** '+ str(e))

        try:
            unmatched_total_c4 = os.path.join(self.params['dirs']['cached_files_folder'], f'{self.client_name}_mismatches_post_flan.parquet')
            local_path = unmatched_total_c4
            s3_path = get_s3_path(local_path, client_name, run_date, processed=False)
            if os.path.exists(local_path)==False and s3_object_exists(s3_path):
                s3_download_file(s3_path, local_path)
            if os.path.exists(unmatched_total_c4) and override==False:
                df_total5 = pd.read_parquet(unmatched_total_c4)
                #s3_upload_data(unmatched_total_c4, client_name, run_date, processed=False)
                
            else:
                log.info('***** Getting Brand for unmatched data*****')
                if self.cache_file_path_dic['sku_p4'] is not None:
                    log.info("************* Adding data from cache *************")
                    local = os.path.join(self.params['dirs']['cached_files_folder'], f'{self.client_name}_mismatches_post_flan_old.parquet')
                    s3_download_file(self.cache_file_path_dic['sku_p4'][1], local)
                    df_old1 = pd.read_parquet(local)
                    df_old1 = df_old1.loc[:, ~df_old1.columns.str.contains('^Unnamed')]
                    relevant_cols = ['pdb_id', 'llm_brand']
                    df_old1 = df_old1.dropna(subset = ['pdb_id']).reset_index(drop = True)
                    df_req = df_total4.loc[~df_total4['pdb_id'].isin(df_old1['pdb_id'].to_list()), :].reset_index(drop=True)
                    df_cache = df_total4.loc[df_total4['pdb_id'].isin(df_old1['pdb_id'].to_list()), :].reset_index(drop=True)
                    log.info(f"columns to be added from cache: {set(df_old1.columns) - set(df_cache.columns)}")
                    df_cache = pd.merge(df_cache, df_old1.loc[:,relevant_cols], on = 'pdb_id', how = 'left')
                    list_cols = ['llm_brand']
                    log.info(f"cache found for {len(df_cache)} entries, computing for {len(df_req)} entries")
                    for col in list_cols:
                        df_cache[col] = df_cache[col].apply(lambda x: x.tolist())
                    if len(df_req)>0:
                        df_req_op = get_brand_multipool_with_retry(df_req, self.client_name)   
                        df_total5 = pd.concat([df_cache, df_req_op], ignore_index=True)
                        df_total5 = df_total5.sort_values(by =['id']).reset_index(drop=True)
                    else:
                        df_total5 = df_cache.sort_values(by =['id']).reset_index(drop=True)
                else:  
                    df_total5 = get_brand_multipool_with_retry(df_total4, self.client_name)   

                df_total5.to_parquet(unmatched_total_c4, index = False)
                s3_upload_data(unmatched_total_c4, client_name, run_date, processed=False)

            #log.info("df_total5 columns", df_total5.columns)
        except Exception as e:
            log.info('***** Error in Getting Brand for unmatched data *****')
            log.info(e)
            raise Exception('***** Error in Getting Brand for unmatched data ***** '+ str(e))
            
        try:
            unmatched_total_c5 = os.path.join(self.params['dirs']['cached_files_folder'], f'{self.client_name}_mismatches_post_llama.parquet')
            local_path = unmatched_total_c5
            s3_path = get_s3_path(local_path, client_name, run_date, processed=False)
            
            if os.path.exists(local_path)==False and s3_object_exists(s3_path):
                s3_download_file(s3_path, local_path)
            if os.path.exists(unmatched_total_c5) and override==False:
                df_total6 = pd.read_parquet(unmatched_total_c5)
                #s3_upload_data(unmatched_total_c5, client_name, run_date, processed=False)
                feature_added = False
                for feature in self.params['llama_features']:
                    if f'llm_{feature}' not in df_total6.columns:
                        feature_added= True
                        df_total6 = get_feature_multipool_llama(df_total6.copy(), feature, self.client_name)  
                if feature_added:
                    df_total6.to_parquet(unmatched_total_c5, index = False)
                    s3_upload_data(unmatched_total_c5, client_name, run_date, processed=False)
            else:
                log.info('***** Getting LLAMA based features for unmatched data*****')
                if self.cache_file_path_dic['sku_p5'] is not None:
                    log.info("************* Adding data from cache *************")
                    local = os.path.join(self.params['dirs']['cached_files_folder'], f'{self.client_name}_mismatches_post_llama_old.parquet')
                    s3_download_file(self.cache_file_path_dic['sku_p5'][1], local)
                    df_old1 = pd.read_parquet(local)
                    df_old1 = df_old1.loc[:, ~df_old1.columns.str.contains('^Unnamed')]
                    llama_cols = [f'llm_{x}' for x in self.params['llama_features']]
                    relevant_cols = ['pdb_id'] + llama_cols
                    df_old1 = df_old1.dropna(subset = ['pdb_id']).reset_index(drop = True)
                    df_req = df_total5.loc[~df_total5['pdb_id'].isin(df_old1['pdb_id'].to_list()), :].reset_index(drop=True)
                    df_cache = df_total5.loc[df_total5['pdb_id'].isin(df_old1['pdb_id'].to_list()), :].reset_index(drop=True)
                    log.info(f"columns to be added from cache: {set(df_old1.columns) - set(df_cache.columns)}")
                    df_cache = pd.merge(df_cache, df_old1.loc[:,relevant_cols], on = 'pdb_id', how = 'left')
                    list_cols = []
                    log.info(f"cache found for {len(df_cache)} entries, computing for {len(df_req)} entries")
                    for col in list_cols:
                        df_cache[col] = df_cache[col].apply(lambda x: x.tolist())
                    if len(df_req)>0:
                        df_req_op = df_req.copy()
                        for feature in self.params['llama_features']:
                              df_req_op = get_feature_multipool_llama(df_req.copy(), feature, self.client_name)
                        df_total6 = pd.concat([df_cache, df_req_op], ignore_index=True)
                        df_total6 = df_total6.sort_values(by =['id']).reset_index(drop=True)
                    else:
                        df_total6 = df_cache.sort_values(by =['id']).reset_index(drop=True)
                else:  
                    df_total6 = df_total5.copy()
                    for feature in self.params['llama_features']:
                          df_total6 = get_feature_multipool_llama(df_total6.copy(), feature, self.client_name)  
                      
                df_total6.to_parquet(unmatched_total_c5, index = False)
                s3_upload_data(unmatched_total_c5, client_name, run_date, processed=False)
        except Exception as e:
            log.info('***** Error in Getting LLAMA based features for unmatched data *****')
            log.info(e)
            raise Exception('***** Error in Getting LLAMA based features for unmatched data ***** '+ str(e))
        
        try:
            unmatched_total_c6 = os.path.join(self.params['dirs']['cached_files_folder'], f'{self.client_name}_mismatches_post_ocr.parquet')
            local_path = unmatched_total_c6
            s3_path = get_s3_path(local_path, client_name, run_date, processed=False)
            if os.path.exists(local_path)==False and s3_object_exists(s3_path):
                s3_download_file(s3_path, local_path)
            if os.path.exists(unmatched_total_c6) and override==False:
                df_total7 = pd.read_parquet(unmatched_total_c6)
                #s3_upload_data(unmatched_total_c6, client_name, run_date, processed=False)
            else:
                log.info('***** Getting OCR for unmatched data*****')
                if self.cache_file_path_dic['sku_p6'] is not None:
                    log.info("************* Adding data from cache *************")
                    local = os.path.join(self.params['dirs']['cached_files_folder'], f'{self.client_name}_mismatches_post_ocr_old.parquet')
                    s3_download_file(self.cache_file_path_dic['sku_p6'][1], local)
                    df_old1 = pd.read_parquet(local)
                    df_old1 = df_old1.loc[:, ~df_old1.columns.str.contains('^Unnamed')]
                    relevant_cols = ['pdb_id', 'ocr_res', 'ocr_res_clean']
                    df_old1 = df_old1.dropna(subset = ['pdb_id']).reset_index(drop = True)
                    df_req = df_total6.loc[~df_total6['pdb_id'].isin(df_old1['pdb_id'].to_list()), :].reset_index(drop=True)
                    df_cache = df_total6.loc[df_total6['pdb_id'].isin(df_old1['pdb_id'].to_list()), :].reset_index(drop=True)
                    log.info(f"columns to be added from cache: {set(df_old1.columns) - set(df_cache.columns)}")
                    df_cache = pd.merge(df_cache, df_old1.loc[:,relevant_cols], on = 'pdb_id', how = 'left')
                    list_cols = []
                    log.info(f"cache found for {len(df_cache)} entries, computing for {len(df_req)} entries")
                    for col in list_cols:
                        df_cache[col] = df_cache[col].apply(lambda x: x.tolist())
                    if len(df_req)>0:
                        image_dat_view = df_req[['fail', 'image']].dropna().copy()
                        if len(image_dat_view)>0:
                            image_data, object_name = prepare_image_data_and_upload_to_s3(self.client_name, df_req, self.run_date)
                            if len(image_data)>0:
                                cache_file_path = os.path.join(PROCESSED_FOLDER_S3, 
                                                                self.client_name, OCR_FOLDER, 
                                                                f'{self.client_name}_ocr_res.parquet')
                                output = generate_OCR_results(object_name, self.client_name, 
                                                                bucket = BUCKET, 
                                                                cache_file_path = cache_file_path)
                                check_ocr_status(output)
                                s3_image_output_path = object_name.split('.')
                                s3_image_output_path = s3_image_output_path[0] + '_output' + '.csv'
                                local_image_output_path = os.path.join(self.params['dirs']['cached_files_folder'], 
                                                                        os.path.split(s3_image_output_path)[1])
                                s3_download_file(s3_image_output_path, local_image_output_path)
                                df_ocr = pd.read_csv(local_image_output_path)
                                df_ocr2 = improve_ocr_data(df_ocr)
                                df_ocr2 = df_ocr2.rename(columns = {'text_original':'ocr_res', 'text_clean':'ocr_res_clean'})
                                df_req_op = pd.merge(df_req, df_ocr2[['image', 'ocr_res', 'ocr_res_clean']], how = 'left', on = 'image')
                                df_req_op.loc[df_req_op['ocr_res'].isna(),'ocr_res'] = None
                                df_req_op.loc[df_req_op['ocr_res_clean'].isna(),'ocr_res_clean'] = None
                                df_req_op.loc[~df_req_op['ocr_res'].isna(),'ocr_res'] = df_req_op.loc[~df_req_op['ocr_res'].isna(),'ocr_res'].astype(str)
                                df_req_op.loc[~df_req_op['ocr_res_clean'].isna(),'ocr_res_clean'] = df_req_op.loc[~df_req_op['ocr_res_clean'].isna(),'ocr_res_clean'].astype(str)
                            else:
                                df_req_op = df_req.copy()
                                df_req_op['ocr_res'] = None
                                df_req_op['ocr_res_clean'] = None
                        else:
                            df_req_op = df_req.copy()
                            df_req_op['ocr_res'] = None
                            df_req_op['ocr_res_clean'] = None
                        df_total7 = pd.concat([df_cache, df_req_op], ignore_index=True)
                        df_total7 = df_total7.sort_values(by =['id']).reset_index(drop=True)
                    else:
                        df_total7 = df_cache.sort_values(by =['id']).reset_index(drop=True)
                else:  

                    image_dat_view = df_total6[['fail', 'image']].dropna().copy()
                    if len(image_dat_view)>0:
                        image_data, object_name = prepare_image_data_and_upload_to_s3(self.client_name, df_total6, self.run_date)
                        if len(image_data)>0:
                            cache_file_path = os.path.join(PROCESSED_FOLDER_S3, 
                                                            self.client_name, OCR_FOLDER, 
                                                            f'{self.client_name}_ocr_res.parquet')
                            output = generate_OCR_results(object_name, self.client_name, 
                                                            bucket = BUCKET, 
                                                            cache_file_path = cache_file_path)
                            check_ocr_status(output)
                            s3_image_output_path = object_name.split('.')
                            s3_image_output_path = s3_image_output_path[0] + '_output' + '.csv'
                            local_image_output_path = os.path.join(self.params['dirs']['cached_files_folder'], 
                                                                    os.path.split(s3_image_output_path)[1])
                            s3_download_file(s3_image_output_path, local_image_output_path)
                            df_ocr = pd.read_csv(local_image_output_path)
                            df_ocr2 = improve_ocr_data(df_ocr)
                            df_ocr2 = df_ocr2.rename(columns = {'text_original':'ocr_res', 'text_clean':'ocr_res_clean'})
                            df_total7 = pd.merge(df_total6, df_ocr2[['image', 'ocr_res', 'ocr_res_clean']], how = 'left', on = 'image')
                            df_total7.loc[df_total7['ocr_res'].isna(),'ocr_res'] = None
                            df_total7.loc[df_total7['ocr_res_clean'].isna(),'ocr_res_clean'] = None
                            df_total7.loc[~df_total7['ocr_res'].isna(),'ocr_res'] = df_total7.loc[~df_total7['ocr_res'].isna(),'ocr_res'].astype(str)
                            df_total7.loc[~df_total7['ocr_res_clean'].isna(),'ocr_res_clean'] = df_total7.loc[~df_total7['ocr_res_clean'].isna(),'ocr_res_clean'].astype(str)
                        else:
                            df_total7 = df_total6.copy()
                            df_total7['ocr_res'] = None
                            df_total7['ocr_res_clean'] = None
                    else:
                        df_total7 = df_total6.copy()
                        df_total7['ocr_res'] = None
                        df_total7['ocr_res_clean'] = None

                df_total7.to_parquet(unmatched_total_c6, index = False)
                s3_upload_data(unmatched_total_c6, client_name, run_date, processed=False)
        except Exception as e:
            log.info('***** Error in Getting OCR for unmatched data *****')
            log.info(e)
            raise Exception('***** Error in Getting OCR for unmatched data ***** '+ str(e))
           

        try:
            unmatched_total_c7 = os.path.join(self.params['dirs']['processed_files_folder'], f'{self.client_name}_processed_mismatches.parquet')
            local_path = unmatched_total_c7
            s3_path = get_s3_path(local_path, client_name, run_date, processed=True)
            if os.path.exists(local_path)==False and s3_object_exists(s3_path):
                s3_download_file(s3_path, local_path)
            if os.path.exists(unmatched_total_c7) and override==False:
                df_total8 = pd.read_parquet(unmatched_total_c7)
                df_total8['ocr_res'] = df_total8['ocr_res'].apply(lambda x: ast.literal_eval(x) if x is not None else x)
                df_total8['ner_out_2'] = df_total8['ner_out_2'].apply(lambda x: ast.literal_eval(x) if x is not None else x)
                df_total8['ner_out'] = df_total8['ner_out'].apply(lambda x: ast.literal_eval(x) if x is not None else x)
                #s3_upload_data(unmatched_total_c7, client_name, run_date, processed=True)
            else:
                log.info('***** Adding weight info for unmatched data*****')
                
                df_total8 = df_total7.copy()
                
                try:
                    df_total8['ocr_res'] = np.vectorize(structure_ocr)(df_total8['ocr_res'])
                except Exception as e:
                    log.info(f'Error in structuring OCR {e}')
                    try:
                        df_total8['ocr_res'] = df_total8['ocr_res'].apply(lambda x: structure_ocr(x))
                    except Exception as e:
                        log.info(f'Error in structuring OCR {e}')
                        raise e
                #df_total8['ocr_res'] = df_total8['ocr_res'].mapply(lambda x: structure_ocr(x))
                
                try:
                    df_total8['ner_out'] = np.vectorize(remove_wt_based_entries)(df_total8['ner_out'])
                except Exception as e:
                    log.info(f'Error in structuring OCR {e}')
                    try:
                        df_total8['ner_out'] = df_total8['ner_out'].apply(lambda x: remove_wt_based_entries(x))
                    except Exception as e:
                        log.info(f'Error in structuring OCR {e}')
                        raise e
                #df_total8['ner_out'] = df_total8['ner_out'].mapply(lambda x: remove_wt_based_entries(x))
                
                
                try:
                    df_total8['ner_out'] = np.vectorize(add_weight_info)(df_total8['ner_out'], df_total8['weight'], df_total8['product_name'])
                except Exception as e:
                    log.info(f'Error in structuring OCR {e}')
                    try:
                        df_total8['ner_out'] = df_total8.apply(lambda x: add_weight_info(x['ner_out'], x['weight'], x['product_name']), axis =1)
                    except Exception as e:
                        log.info(f'Error in structuring OCR {e}')
                        raise e
                #df_total8['ner_out'] = df_total8.mapply(lambda x: add_weight_info(x['ner_out'], x['weight'], x['product_name']), axis =1)
                
                try:
                    df_total8['ner_out_2'] = np.vectorize(add_weight_info_from_ocr)(df_total8['ner_out'], df_total8['ocr_res'])
                except Exception as e:
                    log.info(f'Error in structuring OCR {e}')
                    try:
                        df_total8['ner_out_2'] = df_total8.apply(lambda x: add_weight_info_from_ocr(x['ner_out'], x['ocr_res']), axis = 1)
                    except Exception as e:
                        log.info(f'Error in structuring OCR {e}')
                        raise e
                #df_total8['ner_out_2'] = df_total8.mapply(lambda x: add_weight_info_from_ocr(x['ner_out'], x['ocr_res']), axis = 1)
                df_total8.loc[df_total8['ner_out'].apply(lambda x: len(x)==0),'ner_out'] = df_total8.loc[df_total8['ner_out'].apply(lambda x: len(x)==0),'ner_out_2']     
                df_total8['ner_out'] = df_total8['ner_out'].apply(lambda x: list(x) if type(x)!=list else x)
                df_total8['ner_out_2'] = df_total8['ner_out_2'].apply(lambda x: list(x) if type(x)!=list else x)
                df_total8['ner_out'] = df_total8['ner_out'].apply(lambda x: str(x) if x is not None else x)
                df_total8['ner_out_2'] = df_total8['ner_out_2'].apply(lambda x: str(x) if x is not None else x)
                df_total8['ocr_res'] = df_total8['ocr_res'].apply(lambda x: str(x) if x is not None else x) 
                df_total8.to_parquet(unmatched_total_c7, index = False)
                s3_upload_data(unmatched_total_c7, client_name, run_date, processed=True)
                      
        except Exception as e:
            log.info('***** Error in Adding weight info for unmatched data *****')
            log.info(e)
            raise Exception('***** Error in Adding weight info for unmatched data ***** '+ str(e))

        column_set = set()
        new_columns = set()
        #================================ Unmatched data processing complete ==================================        
        #======================================= process PIM data =============================================
        try:
            pim_c1 = os.path.join(self.params['dirs']['cached_files_folder'], f'{self.client_name}_post_ner_pim.parquet')
            local_path = pim_c1
            s3_path = get_s3_path(local_path, client_name, run_date, processed=False)
            if os.path.exists(local_path)==False and s3_object_exists(s3_path):
                s3_download_file(s3_path, local_path)
            if os.path.exists(pim_c1) and override==False:
                df_pim2 = pd.read_parquet(pim_c1)
                #s3_upload_data(pim_c1, client_name, run_date, processed=False)
            else:
                self.df_pim = self.df_pim.reset_index(drop=True)
                log.info('***** Cleaning Titles for PIM data *****')   
                df_pim1 = self.df_pim.copy()
                if self.cache_file_path_dic['pim_p1'] is not None:
                    log.info("************* Adding data from cache *************")
                    local = os.path.join(self.params['dirs']['cached_files_folder'], f'{self.client_name}_post_ner_pim_old.parquet')
                    s3_download_file(self.cache_file_path_dic['pim_p1'][1], local)
                    df_old_pim = pd.read_parquet(local)
                    df_old_pim = df_old_pim.loc[:, ~df_old_pim.columns.str.contains('^Unnamed')]
                    relevant_cols = ['gtin', 'name_cleaned', 'ner_out', 'ner_out_updated', 'size_regex']
                    df_old_pim = df_old_pim.dropna(subset = ['name_cleaned', 'gtin']).reset_index(drop = True)
                    df_req = df_pim1.loc[~df_pim1['gtin'].isin(df_old_pim['gtin'].to_list()), :].reset_index(drop=True)
                    df_cache = df_pim1.loc[df_pim1['gtin'].isin(df_old_pim['gtin'].to_list()), :].reset_index(drop=True)
                    log.info(f"columns to be added from cache: {set(df_old_pim.columns) - set(df_cache.columns)}")
                    df_cache = pd.merge(df_cache, df_old_pim.loc[:,relevant_cols], on = 'gtin', how = 'left')
                    list_cols = ['ner_out', 'ner_out_updated']
                    log.info(f"cache found for {len(df_cache)} entries, computing for {len(df_req)} entries")
                    for col in list_cols:
                        df_cache[col] = df_cache[col].apply(lambda x: x.tolist())
                    if len(df_req)>0:
                        df_req_op = get_cleaned_title(df_req.copy(), client_name)
                        df_req_op['size_regex'] = df_req_op['ner_out_updated'].map(get_size_regex)
                        for i in range(len(df_req_op)):
                            p = re.search(r'(\d+(?:\.\d+)?)\s*-?\s*(fz)', df_req_op.loc[i, 'product_name'], re.IGNORECASE)
                            if p!=None:
                                df_req_op.loc[i, 'product_name'] = re.sub(r'(\d+(?:\.\d+)?)\s*-?\s*(fz)', f'{p.group(1)} oz', df_req_op.loc[i, 'product_name'], flags=re.I)
                        df_pim2 = pd.concat([df_cache, df_req_op], ignore_index=True)
                        df_pim2 = df_pim2.sort_values(by =['id']).reset_index(drop=True)
                    else:
                        df_pim2 = df_cache.sort_values(by =['id']).reset_index(drop=True)
                else:
                    #log.info('***** Cleaning Titles for PIM data *****')
                    df_pim2 = get_cleaned_title(df_pim1.copy(), client_name)
                    df_pim2['size_regex'] = df_pim2['ner_out_updated'].map(get_size_regex)
                    for i in range(len(df_pim2)):
                        p = re.search(r'(\d+(?:\.\d+)?)\s*-?\s*(fz)', df_pim2.loc[i, 'product_name'], re.IGNORECASE)
                        if p!=None:
                            df_pim2.loc[i, 'product_name'] = re.sub(r'(\d+(?:\.\d+)?)\s*-?\s*(fz)', f'{p.group(1)} oz', df_pim2.loc[i, 'product_name'], flags=re.I)
                    df_pim2 = df_pim2.sort_values(by =['id']).reset_index(drop=True)

                df_pim2.to_parquet(pim_c1, index = False)
                s3_upload_data(pim_c1, client_name, run_date, processed=False)


        except Exception as e:
            log.info('***** Error in Cleaning Titles for PIM data *****')
            log.info(e)
            raise Exception('***** Error in Cleaning Titles for PIM data ***** '+ str(e))
        try:
            pim_c2 = os.path.join(self.params['dirs']['cached_files_folder'], f'{self.client_name}_post_images_post_ner_pim.parquet')
            local_path = pim_c2
            s3_path = get_s3_path(local_path, client_name, run_date, processed=False)
            if os.path.exists(local_path)==False and s3_object_exists(s3_path):
                s3_download_file(s3_path, local_path)
            if os.path.exists(pim_c2) and override==False:
                df_pim3 = pd.read_parquet(pim_c2)
                #s3_upload_data(pim_c2, client_name, run_date, processed=False)
            else:
                
                log.info('***** Downloading Images for PIM data *****')
                image_download_occured_l2 = True
                df_pim3 = download_images_create_flag(df_pim2)
                log.info(f"df_pim3 columns {df_pim3.columns}")
                df_pim3 = remove_white_space_all(df_pim3)
                log.info(f"Image download failed for  {df_pim3.loc[df_pim3['fail']==1, 'image_path'].isna().sum()}")
                df_pim3.to_parquet(pim_c2, index = False)
                s3_upload_data(pim_c2, client_name, run_date, processed=False)
            
        except Exception as e:
            image_download_occured_l2 = False
            log.info('***** Error in Downloading Images for PIM data *****')
            log.info(e)
            raise Exception('***** Error in Downloading Images for PIM data ***** '+ str(e))
        try:
            ### create image embeddings for PIM
            local_path = os.path.join(self.params['dirs']['processed_files_folder'], f'{self.client_name}_pim_raw_emb.parquet')
            s3_path = get_s3_path(local_path, client_name, run_date, processed=True)
            if s3_object_exists(s3_path) == False:
                if os.path.exists(os.path.join(self.params['dirs']['processed_files_folder'], f'{self.client_name}_pim_raw_emb.parquet'))== False or override:
                    if image_download_occured_l2 == False:
                        log.info(f"****** Downloading Images for PIM *************")
                        temp = download_images_create_flag(df_pim2)
                        temp2 = remove_white_space_all(temp)
                        image_download_occured_l2 = True

                    create_and_save_image_embeddings(df_pim3, self.params['dirs']['processed_files_folder'], 
                                                     self.client_name, file_type = 'pim', emb_type = 'raw')
                if s3_upload_data(os.path.join(self.params['dirs']['processed_files_folder'], f'{self.client_name}_pim_raw_emb.parquet'), self.client_name, self.run_date, processed=True)==1:
                    log.info("Raw PIM embeddings uploaded successfully")

            local_path = os.path.join(self.params['dirs']['processed_files_folder'], f'{self.client_name}_pim_pro_emb.parquet')
            s3_path = get_s3_path(local_path, client_name, run_date, processed=True)
            if s3_object_exists(s3_path) == False:
                if os.path.exists(os.path.join(self.params['dirs']['processed_files_folder'],f'{self.client_name}_pim_pro_emb.parquet'))== False or override:
                    if image_download_occured_l2 == False:
                        log.info(f"******* Downloading Images for PIM *************")
                        temp = download_images_create_flag(df_pim2)
                        temp2 = remove_white_space_all(temp)
                        image_download_occured_l2 = True
                    create_and_save_image_embeddings(df_pim3, self.params['dirs']['processed_files_folder'], self.client_name, file_type = 'pim', emb_type = 'pro')

                if s3_upload_data(os.path.join(self.params['dirs']['processed_files_folder'], f'{self.client_name}_pim_pro_emb.parquet'), self.client_name, self.run_date, processed=True) ==1:
                    log.info("Pro PIM embeddings uploaded successfully")
        except Exception as e:
            log.info('***** Error in Creating PIM Image Embeddings *****')
            log.info(e)   
            raise Exception('***** Error in Creating PIM Image Embeddings ***** '+ str(e))
        
#         try:
#             pim_c3 = os.path.join(self.params['dirs']['cached_files_folder'], f'{self.client_name}_pim_post_uni_ner.parquet')
#             local_path = pim_c3
#             s3_path = get_s3_path(local_path, client_name, run_date, processed=False)
#             if os.path.exists(local_path)==False and s3_object_exists(s3_path):
#                 s3_download_file(s3_path, local_path)
#             if os.path.exists(pim_c3) and override==False:
#                 df_pim4 = pd.read_parquet(pim_c3)
#                 #s3_upload_data(pim_c3, client_name, run_date, processed=False)
#             else:
                
#                 log.info('***** Getting NER for PIM data*****')
#                 if self.cache_file_path_dic['pim_p3'] is not None:
#                     log.info("************* Adding data from cache *************")
#                     local = os.path.join(self.params['dirs']['cached_files_folder'], f'{self.client_name}_pim_post_uni_ner_old.parquet')
#                     s3_download_file(self.cache_file_path_dic['pim_p3'][1], local)
#                     df_old_pim = pd.read_parquet(local)
#                     df_old_pim = df_old_pim.loc[:, ~df_old_pim.columns.str.contains('^Unnamed')]
#                     relevant_cols = ['gtin', 'uni_ner_brand']
#                     df_old_pim = df_old_pim.dropna(subset = ['gtin']).reset_index(drop = True)
#                     df_req = df_pim3.loc[~df_pim3['gtin'].isin(df_old_pim['gtin'].to_list()), :].reset_index(drop=True)
#                     df_cache = df_pim3.loc[df_pim3['gtin'].isin(df_old_pim['gtin'].to_list()), :].reset_index(drop=True)
#                     log.info(f"columns to be added from cache: {set(df_old_pim.columns) - set(df_cache.columns)}")
#                     df_cache = pd.merge(df_cache, df_old_pim.loc[:,relevant_cols], on = 'gtin', how = 'left')
#                     list_cols = ['uni_ner_brand']
#                     log.info(f"cache found for {len(df_cache)} entries, computing for {len(df_req)} entries")
#                     for col in list_cols:
#                         df_cache[col] = df_cache[col].apply(lambda x: x.tolist())
#                     if len(df_req)>0:
#                         uniner = UniNER(self.client_name)
#                         df_req_op = uniner.get_brand_multipool(df_req.copy())
#                         df_pim4 = pd.concat([df_cache, df_req_op], ignore_index=True)
#                         df_pim4 = df_pim4.sort_values(by =['id']).reset_index(drop=True)
#                     else:
#                         df_pim4 = df_cache.sort_values(by =['id']).reset_index(drop=True)
#                 else:
#                     uniner = UniNER(self.client_name)
#                     df_pim4 = df_pim3.copy()
#                     #for feature in self.params['uni_ner_features']:
#                         #df_pim4 = uniner.get_feature_multipool(df_pim4.copy(), feature) 
#                     df_pim4 = uniner.get_brand_multipool(df_pim4.copy())
#                 df_pim4.to_parquet(pim_c3, index = False)
#                 s3_upload_data(pim_c3, client_name, run_date, processed=False)
                
#         except Exception as e:
#             log.info('***** Error in Getting NER for pim data *****')
#             log.info(e)
#             raise Exception('***** Error in Getting NER for pim data ***** '+ str(e))

        df_pim4 = df_pim3
        try:
            pim_c4 = os.path.join(self.params['dirs']['cached_files_folder'], f'{self.client_name}_pim_post_flan.parquet')
            local_path = pim_c4
            s3_path = get_s3_path(local_path, client_name, run_date, processed=False)
            if os.path.exists(local_path)==False and s3_object_exists(s3_path):
                s3_download_file(s3_path, local_path)
            if os.path.exists(pim_c4) and override==False:
                df_pim5 = pd.read_parquet(pim_c4)
                #s3_upload_data(pim_c4, client_name, run_date, processed=False)
            else:
                log.info('***** Getting Brand for PIM data*****')
                if self.cache_file_path_dic['pim_p4'] is not None:
                    log.info("************* Adding data from cache *************")
                    local = os.path.join(self.params['dirs']['cached_files_folder'], f'{self.client_name}_pim_post_flan.parquet')
                    s3_download_file(self.cache_file_path_dic['pim_p4'][1], local)
                    df_old_pim = pd.read_parquet(local)
                    df_old_pim = df_old_pim.loc[:, ~df_old_pim.columns.str.contains('^Unnamed')]
                    relevant_cols = ['gtin', 'llm_brand']
                    df_old_pim = df_old_pim.dropna(subset = ['gtin']).reset_index(drop = True)
                    df_req = df_pim4.loc[~df_pim4['gtin'].isin(df_old_pim['gtin'].to_list()), :].reset_index(drop=True)
                    df_cache = df_pim4.loc[df_pim4['gtin'].isin(df_old_pim['gtin'].to_list()), :].reset_index(drop=True)
                    log.info(f"columns to be added from cache: {set(df_old_pim.columns) - set(df_cache.columns)}")
                    df_cache = pd.merge(df_cache, df_old_pim.loc[:,relevant_cols], on = 'gtin', how = 'left')
                    list_cols = ['llm_brand']
                    log.info(f"cache found for {len(df_cache)} entries, computing for {len(df_req)} entries")
                    for col in list_cols:
                        df_cache[col] = df_cache[col].apply(lambda x: x.tolist())
                    if len(df_req)>0:
                        df_req_op = get_brand_multipool_with_retry(df_req, self.client_name)   
                        df_pim5 = pd.concat([df_cache, df_req_op], ignore_index=True)
                        df_pim5 = df_pim5.sort_values(by =['id']).reset_index(drop=True)
                    else:
                        df_pim5 = df_cache.sort_values(by =['id']).reset_index(drop=True)
                else:  
                    df_pim5 = get_brand_multipool_with_retry(df_pim4, self.client_name)   

                df_pim5.to_parquet(pim_c4, index = False)
                s3_upload_data(pim_c4, client_name, run_date, processed=False)

        except Exception as e:
            log.info('***** Error in Getting Brand for pim data *****')
            log.info(e)
            raise Exception('***** Error in Getting Brand for pim data ***** '+ str(e))
            
        try:
            pim_c5 = os.path.join(self.params['dirs']['cached_files_folder'], f'{self.client_name}_pim_post_llama.parquet')
            local_path = pim_c5
            s3_path = get_s3_path(local_path, client_name, run_date, processed=False)
            if os.path.exists(local_path)==False and s3_object_exists(s3_path):
                s3_download_file(s3_path, local_path)
            if os.path.exists(pim_c5) and override==False:
                df_pim6 = pd.read_parquet(pim_c5)
                #s3_upload_data(pim_c4, client_name, run_date, processed=False)
            else:
                log.info('***** Getting LLAMA based features for PIM data*****')
                if self.cache_file_path_dic['pim_p5'] is not None:
                    log.info("************* Adding data from cache *************")
                    local = os.path.join(self.params['dirs']['cached_files_folder'], f'{self.client_name}_pim_post_llama_old.parquet')
                    s3_download_file(self.cache_file_path_dic['pim_p5'][1], local)
                    df_old1 = pd.read_parquet(local)
                    df_old1 = df_old1.loc[:, ~df_old1.columns.str.contains('^Unnamed')]
                    llama_cols = [f'llm_{x}' for x in self.params['llama_features']]
                    relevant_cols = ['gtin'] + llama_cols
                    df_old1 = df_old1.dropna(subset = ['gtin']).reset_index(drop = True)
                    df_req = df_pim5.loc[~df_pim5['gtin'].isin(df_old1['gtin'].to_list()), :].reset_index(drop=True)
                    df_cache = df_pim5.loc[df_pim5['gtin'].isin(df_old1['gtin'].to_list()), :].reset_index(drop=True)
                    log.info(f"columns to be added from cache: {set(df_old1.columns) - set(df_cache.columns)}")
                    df_cache = pd.merge(df_cache, df_old1.loc[:,relevant_cols], on = 'gtin', how = 'left')
                    list_cols = []
                    log.info(f"cache found for {len(df_cache)} entries, computing for {len(df_req)} entries")
                    for col in list_cols:
                        df_cache[col] = df_cache[col].apply(lambda x: x.tolist())
                    if len(df_req)>0:
                        df_req_op = df_req.copy()
                        for feature in self.params['llama_features']:
                                df_req_op = get_feature_multipool_llama(df_req.copy(), feature, self.client_name)
                        df_pim6 = pd.concat([df_cache, df_req_op], ignore_index=True)
                        df_pim6 = df_pim6.sort_values(by =['id']).reset_index(drop=True)
                    else:
                        df_pim6 = df_cache.sort_values(by =['id']).reset_index(drop=True)
                else:  
                    df_pim6 = df_pim5.copy()
                    for feature in self.params['llama_features']:
                            df_pim6 = get_feature_multipool_llama(df_pim6.copy(), feature, self.client_name)  

                df_pim6.to_parquet(pim_c5, index = False)
                s3_upload_data(pim_c5, client_name, run_date, processed=False)
            
        except Exception as e:
            log.info('***** Error in Getting (LLAMA) for pim data *****')
            log.info(e)
            raise Exception('***** Error in Getting (LLAMA) for pim data ***** '+ str(e))
          
        
        try:
            pim_c6 = os.path.join(self.params['dirs']['cached_files_folder'], f'{self.client_name}_pim_post_ocr.parquet')
            local_path = pim_c6
            s3_path = get_s3_path(local_path, client_name, run_date, processed=False)
            if os.path.exists(local_path)==False and s3_object_exists(s3_path):
                s3_download_file(s3_path, local_path)
            if os.path.exists(pim_c6) and override==False:
                df_pim7 = pd.read_parquet(pim_c6)
                #s3_upload_data(pim_c6, client_name, run_date, processed=False)
            else:
                log.info('***** Getting OCR for PIM data*****')
                if self.cache_file_path_dic['pim_p6'] is not None:
                    log.info("************* Adding data from cache *************")
                    local = os.path.join(self.params['dirs']['cached_files_folder'], f'{self.client_name}_pim_post_ocr_old.parquet')
                    s3_download_file(self.cache_file_path_dic['pim_p6'][1], local)
                    df_old_pim = pd.read_parquet(local)
                    relevant_cols = ['gtin', 'ocr_res', 'ocr_res_clean']
                    df_old_pim = df_old_pim.dropna(subset = ['gtin']).reset_index(drop = True)
                    df_req = df_pim6.loc[~df_pim6['gtin'].isin(df_old_pim['gtin'].to_list()), :].reset_index(drop=True)
                    df_cache = df_pim6.loc[df_pim6['gtin'].isin(df_old_pim['gtin'].to_list()), :].reset_index(drop=True)
                    log.info(f"columns to be added from cache: {set(df_old_pim.columns) - set(df_cache.columns)}")
                    df_cache = pd.merge(df_cache, df_old_pim.loc[:,relevant_cols], on = 'gtin', how = 'left')
                    list_cols = []
                    log.info(f"cache found for {len(df_cache)} entries, computing for {len(df_req)} entries")
                    for col in list_cols:
                        df_cache[col] = df_cache[col].apply(lambda x: x.tolist())
                    if len(df_req)>0:
                        image_dat_view = df_req[['fail', 'image']].dropna().copy()
                        if len(image_dat_view)>0:
                            image_data, object_name = prepare_image_data_and_upload_to_s3(self.client_name, df_req, self.run_date)
                            if len(image_data)>0:
                                cache_file_path = os.path.join(PROCESSED_FOLDER_S3, 
                                                                self.client_name, OCR_FOLDER, 
                                                                f'{self.client_name}_ocr_res.parquet')
                                output = generate_OCR_results(object_name, self.client_name, 
                                                                bucket = BUCKET, 
                                                                cache_file_path = cache_file_path)
                                check_ocr_status(output)
                                s3_image_output_path = object_name.split('.')
                                s3_image_output_path = s3_image_output_path[0] + '_output' + '.csv'
                                local_image_output_path = os.path.join(self.params['dirs']['cached_files_folder'], 
                                                                        os.path.split(s3_image_output_path)[1])
                                s3_download_file(s3_image_output_path, local_image_output_path)
                                df_ocr = pd.read_csv(local_image_output_path)
                                df_ocr2 = improve_ocr_data(df_ocr)
                                df_ocr2 = df_ocr2.rename(columns = {'text_original':'ocr_res', 'text_clean':'ocr_res_clean'})
                                df_req_op = pd.merge(df_req, df_ocr2[['image', 'ocr_res', 'ocr_res_clean']], how = 'left', on = 'image')
                                df_req_op.loc[df_req_op['ocr_res'].isna(),'ocr_res'] = None
                                df_req_op.loc[df_req_op['ocr_res_clean'].isna(),'ocr_res_clean'] = None
                                df_req_op.loc[~df_req_op['ocr_res'].isna(),'ocr_res'] = df_req_op.loc[~df_req_op['ocr_res'].isna(),'ocr_res'].astype(str)
                                df_req_op.loc[~df_req_op['ocr_res_clean'].isna(),'ocr_res_clean'] = df_req_op.loc[~df_req_op['ocr_res_clean'].isna(),'ocr_res_clean'].astype(str)
                            else:
                                df_req_op = df_req.copy()
                                df_req_op['ocr_res'] = None
                                df_req_op['ocr_res_clean'] = None
                        else:
                            df_req_op = df_req.copy()
                            df_req_op['ocr_res'] = None
                            df_req_op['ocr_res_clean'] = None
                        df_pim7 = pd.concat([df_cache, df_req_op], ignore_index=True)
                        df_pim7 = df_pim7.sort_values(by =['id']).reset_index(drop=True)
                    else:
                        df_pim7 = df_cache.sort_values(by =['id']).reset_index(drop=True)
                else:  

                    image_dat_view = df_pim6[['fail', 'image']].dropna().copy()
                    if len(image_dat_view)>0:
                        image_data, object_name = prepare_image_data_and_upload_to_s3(self.client_name, df_pim6, self.run_date, pim = True)
                        if len(image_data)>0:
                            cache_file_path = os.path.join(PROCESSED_FOLDER_S3, 
                                                            self.client_name, OCR_FOLDER, 
                                                            f'{self.client_name}_ocr_res.parquet')
                            output = generate_OCR_results(object_name, self.client_name, 
                                                            bucket = BUCKET, 
                                                            cache_file_path = cache_file_path)
                            check_ocr_status(output)
                            s3_image_output_path = object_name.split('.')
                            s3_image_output_path = s3_image_output_path[0] + '_output' + '.csv'
                            local_image_output_path = os.path.join(self.params['dirs']['cached_files_folder'], 
                                                                    os.path.split(s3_image_output_path)[1])
                            s3_download_file(s3_image_output_path, local_image_output_path)
                            df_ocr = pd.read_csv(local_image_output_path)
                            df_ocr2 = improve_ocr_data(df_ocr)
                            df_ocr2 = df_ocr2.rename(columns = {'text_original':'ocr_res', 'text_clean':'ocr_res_clean'})
                            df_pim7 = pd.merge(df_pim6, df_ocr2[['image', 'ocr_res', 'ocr_res_clean']], how = 'left', on = 'image')
                            df_pim7.loc[df_pim7['ocr_res'].isna(),'ocr_res'] = None
                            df_pim7.loc[df_pim7['ocr_res_clean'].isna(),'ocr_res_clean'] = None
                            df_pim7.loc[~df_pim7['ocr_res'].isna(),'ocr_res'] = df_pim7.loc[~df_pim7['ocr_res'].isna(),'ocr_res'].astype(str)
                            df_pim7.loc[~df_pim7['ocr_res_clean'].isna(),'ocr_res_clean'] = df_pim7.loc[~df_pim7['ocr_res_clean'].isna(),'ocr_res_clean'].astype(str)
                        else:
                            df_pim7 = df_pim6.copy()
                            df_pim7['ocr_res'] = None
                            df_pim7['ocr_res_clean'] = None
                    else:
                        df_pim7 = df_pim6.copy()
                        df_pim7['ocr_res'] = None
                        df_pim7['ocr_res_clean'] = None

                df_pim7.to_parquet(pim_c6, index = False)
                s3_upload_data(pim_c6, client_name, run_date, processed=False)
        except Exception as e:
            log.info('***** Error in Getting OCR for pim data *****')
            log.info(e)
            raise Exception('***** Error in Getting OCR for pim data ***** '+ str(e))
        

        try:
            pim_c7 = os.path.join(self.params['dirs']['processed_files_folder'], f'{self.client_name}_processed_pim.parquet')
            local_path = pim_c7
            s3_path = get_s3_path(local_path, client_name, run_date, processed=True)
            if os.path.exists(local_path)==False and s3_object_exists(s3_path):
                s3_download_file(s3_path, local_path)
            if os.path.exists(pim_c7) and override==False:
                df_pim8 = pd.read_parquet(pim_c7)
                #s3_upload_data(pim_c7, client_name, run_date, processed=True)
                df_pim8['ocr_res'] = df_pim8['ocr_res'].apply(lambda x: ast.literal_eval(x) if x is not None else x)
                df_pim8['ner_out_2'] = df_pim8['ner_out_2'].apply(lambda x: ast.literal_eval(x) if x is not None else x)
                df_pim8['ner_out'] = df_pim8['ner_out'].apply(lambda x: ast.literal_eval(x) if x is not None else x)
            else:
                
                log.info('***** Adding weight info for pim data*****')
                df_pim8 = df_pim7.copy()
                try:
                    df_pim8['ocr_res'] = np.vectorize(structure_ocr)(df_pim8['ocr_res'])
                except Exception as e:
                    log.info(f'Error in structuring OCR {e}')
                    try:
                        df_pim8['ocr_res'] = df_pim8['ocr_res'].apply(lambda x: structure_ocr(x))
                    except Exception as e:
                        log.info(f'Error in structuring OCR {e}')
                        raise e
                        
                #df_pim8['ocr_res'] = df_pim8['ocr_res'].mapply(lambda x: structure_ocr(x))
                try:
                    df_pim8['ner_out_2'] = np.vectorize(add_weight_info_from_ocr)(df_pim8['ner_out'], df_pim8['ocr_res'])
                except Exception as e:
                    log.info(f'Error in structuring OCR {e}')
                    try:
                        df_pim8['ner_out_2'] = df_pim8.apply(lambda x: add_weight_info_from_ocr(x['ner_out'], x['ocr_res']), axis = 1)
                    except Exception as e:
                        log.info(f'Error in structuring OCR {e}')
                        raise e
                #df_pim8['ner_out_2'] = df_pim8.mapply(lambda x: add_weight_info_from_ocr(x['ner_out'], x['ocr_res']), axis = 1)
                df_pim8.loc[df_pim8['ner_out'].apply(lambda x: len(x)==0),'ner_out'] = df_pim8.loc[df_pim8['ner_out'].apply(lambda x: len(x)==0),'ner_out_2']     
                df_pim8['ner_out'] = df_pim8['ner_out'].apply(lambda x: list(x) if type(x)!=list else x)
                df_pim8['ner_out_2'] = df_pim8['ner_out_2'].apply(lambda x: list(x) if type(x)!=list else x)
                df_pim8['ner_out'] = df_pim8['ner_out'].apply(lambda x: str(x) if x is not None else x)
                df_pim8['ner_out_2'] = df_pim8['ner_out_2'].apply(lambda x: str(x) if x is not None else x)
                df_pim8['ocr_res'] = df_pim8['ocr_res'].apply(lambda x: str(x) if x is not None else x) 
                df_pim8.to_parquet(pim_c7, index = False)
                s3_upload_data(pim_c7, client_name, run_date, processed=True)
            
        except Exception as e:
            log.info('***** Error in Adding weight info for pim data *****')
            log.info(e)  
            raise Exception('***** Error in Adding weight for pim data ***** '+ str(e))
            

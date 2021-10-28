import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString
from shapely.geometry import Polygon
from pathlib import Path
import glob, json , sys
import numpy as np
from pathlib import Path
import sys
import multiprocessing
import matplotlib.pyplot as plt
# import cv2
from itertools import product
import itertools

fldr = "/"+sys.argv[2]+"_"
fpo_name = sys.argv[2]

print(fldr)

def intersecter(ori , other):
    ''' c == json.load final op'''
    ''' takes in raw json and returns new json without overlaps'''
    
    list_ori = ori['features']
    list_oth = other['features']
    
    to_remove_ff = []
    for i in (list_ori):
        for j in (list_oth):

            tmp_i = Polygon(i['geometry']['coordinates'][0])
            tmp_j = Polygon(j['geometry']['coordinates'][0])
            if not tmp_i.is_valid :
                tmp_i = tmp_i.buffer(0)
            if not tmp_j.is_valid :
                tmp_j = tmp_j.buffer(0)
                
                
            if tmp_i.is_valid and tmp_i.is_valid :
            

                try:
                    
                    if tmp_i.intersection(tmp_j).area/(tmp_i.area + 1e-13) * 100 > 60 or tmp_j.intersection(tmp_i).area/(tmp_j.area +  1e-13) * 100 > 60 :
                        to_remove_ff.append((i['id'], j['id']))
                except:
                    to_remove_ff.append((i['id'], j['id']))
                    to_remove_ff.append((i['id'], j['id']))
                        
    
    remove_indices = list(set(to_remove_ff))
    
    return remove_indices



def processor_subtile_remov(l):

        ''' get each 3 elems '''
        i = l[0]
        j = l[1]
        right_elem = i+540, j
        diag_elem = i+540, j+540
        down_elem = i, j+540
        
        orwi, orwj =i, j 
#         print((i,j), right_elem, diag_elem, down_elem)
        
        try:
            ''' check if already present'''

            ''' get the intersection between tiles '''

            name_right = f"{main_folder_path}{fldr}{right_elem[0]}_{right_elem[1]}.geojson"
            name_diag = f"{main_folder_path}{fldr}{diag_elem[0]}_{diag_elem[1]}.geojson"
            name_down = f"{main_folder_path}{fldr}{down_elem[0]}_{down_elem[1]}.geojson"
            name_orig = f"{main_folder_path}{fldr}{i}_{j}.geojson"

            dict_place = {'right': f"{right_elem[0]}_{right_elem[1]}",
                         'diag' : f"{diag_elem[0]}_{diag_elem[1]}",
                         'down' : f"{down_elem[0]}_{down_elem[1]}",
                         'orig' : f"{i}_{j}"}

            ''' declare '''
            orig_diag, orig_right, orig_down = [], [], []
            diag_identify , right_identify, down_identify = False, False , False

            if not Path(name_orig).is_file():
                pass
    #                 print('not done ', (i,j))
            else:

                with open(name_orig) as k :
                    orig = json.load(k)
                    for id_, elem in enumerate(orig['features']):
                        elem['id'] = id_
                        elem['tile_name'] = f"{i}_{j}"



                if Path(name_diag).is_file():
                    diag_identify = True
                    with open(name_diag) as k :
                        diag = json.load(k)
                    for id_, elem in enumerate(diag['features']):
                        elem['id'] = id_

                    orig_diag = intersecter(orig, diag)
                    diag_df = pd.DataFrame(orig_diag)
                    diag_df['place'] = 'diag'

                    diag_full = pd.DataFrame(diag['features'])
                else:
                    diag_identify = False
                    diag_full = pd.DataFrame(columns=['type', 'properties', 'geometry', 'id', 'tile_name', 'place'])

                if Path(name_down).is_file():
                    down_identify = True
                    with open(name_down) as k :
                        down = json.load(k)
                    for id_, elem in enumerate(down['features']):
                        elem['id'] = id_

                    orig_down = intersecter(orig, down)

                    down_df = pd.DataFrame(orig_down)
                    down_df['place'] = 'down'

                    down_full = pd.DataFrame(down['features'])

                else:
                    down_identify = False
                    down_full = pd.DataFrame(columns=['type', 'properties', 'geometry', 'id', 'tile_name', 'place'])


                if Path(name_right).is_file():
                    right_identify = True
                    with open(name_right) as k :
                        right = json.load(k)
                    for id_, elem in enumerate(right['features']):
                        elem['id'] = id_
                    orig_right = intersecter(orig, right)      

                    right_df = pd.DataFrame(orig_right)
                    right_df['place'] = 'right'

                    right_full = pd.DataFrame(right['features'])

                else:
                    right_identify = False
                    right_full = pd.DataFrame(columns=['type', 'properties', 'geometry', 'id', 'tile_name', 'place'])



                if len(orig_diag) > 0:
                    diag_df['geometry'] = diag_df[1].apply(lambda x : diag['features'][x]['geometry']['coordinates'][0])

                if len(orig_right) > 0:
                    right_df['geometry'] = right_df[1].apply(lambda x : right['features'][x]['geometry']['coordinates'][0])

                if len(orig_down) > 0:
                    down_df['geometry'] = down_df[1].apply(lambda x : down['features'][x]['geometry']['coordinates'][0])


                ''' append data '''

                ''' need to handle if all empty'''
                keep = True

                if len(orig_diag) < 1 and len(orig_right) < 1 and len(orig_down) < 1 :
    #                     print('no data')

                    t1 = pd.DataFrame(columns = [0, 1, 'place', 'geometry'])


                elif len(orig_diag) > 0 and len(orig_right) > 0 and len(orig_down) > 0 :

                    t1 = diag_df.append(right_df, ignore_index = True).append(down_df, ignore_index = True)[[0, 1 ,'place', 'geometry']]

                elif len(orig_diag) > 0 and len(orig_right) > 0 and len(orig_down) < 1:

                    t1 = diag_df.append(right_df, ignore_index = True)[[0, 1, 'place', 'geometry']]

                elif len(orig_diag) > 0 and len(orig_down) > 0 and len(orig_right) < 1:

                    t1 = diag_df.append(down_df, ignore_index = True)[[0, 1, 'place', 'geometry']]

                elif len(orig_right) > 0 and len(orig_down) > 0 and len(orig_diag) < 1:

                    t1 = right_df.append(down_df, ignore_index = True)[[0, 1, 'place', 'geometry']]

                else:

                    if len(orig_diag) > 0:

                        t1 = diag_df[[1,0, 'place', 'geometry']]

                    if len(orig_down) > 0:

                        t1 = down_df[[1,0 , 'place', 'geometry']]

                    if len(orig_right) > 0:
                        t1 = right_df[[1,0 , 'place', 'geometry']]


                ''' join back the orig '''

                only_orig_df = pd.DataFrame(t1[0].apply(lambda x : orig['features'][x]['geometry']['coordinates'][0])).rename(columns = {0:'geometry'})
                only_orig_df['place'] = 'orig'
                only_orig_df[0] = t1[0].values

                t1 = t1[[ 0,1,'place', 'geometry']]

                fdf = only_orig_df.append(t1, ignore_index = True)

                fdf['geometry'] = fdf.geometry.apply(lambda x : Polygon(x))

                fdf = gpd.GeoDataFrame(fdf, geometry  = 'geometry').set_crs(epsg=4326, inplace=True)

                merged_ = fdf.dissolve(by=0).reset_index(drop = True)

                or_ = fdf[fdf[1].isna()==True].reset_index(drop = True)
                or_['tile_name'] = or_['place'].map(dict_place)
                or_ = or_[['geometry','tile_name', 'place', 0]]
                or_.rename(columns={0 :  'id'}, inplace = True)

                abc = fdf.dropna().reset_index(drop = True)
                abc['tile_name'] = abc['place'].map(dict_place)
                abc = abc[['geometry','tile_name', 'place', 1]]
                abc.rename(columns={1 :'id'}, inplace = True)

                to_rem = abc.append(or_, ignore_index = True)

                to_rem['id'] = to_rem.id.astype(int)
                to_rem['id'] = to_rem.id.astype(str)


    #             return fdf
                ''' keep and delete operation'''

                orig_full = pd.DataFrame(orig['features'])
                to_drop = fdf[fdf['place'] == 'orig'][0].unique()
                if to_drop.shape[0] >0:
                    orig_full =  orig_full.drop(orig_full.index[to_drop]).reset_index(drop = True)
                else:
                    orig_full =  orig_full

                orig_full['tile_name'] = f'{i}_{j}'
                orig_full['place'] = 'orig'

        #         diag_full = pd.DataFrame(diag['features'])
                #to_drop = fdf[fdf['place'] == 'diag'][0].unique()
                to_drop = [el[1] for el in orig_diag] if len(orig_diag) > 0 else []
                diag_full =  diag_full.drop(diag_full.index[list(set(to_drop))]).reset_index(drop = True)
                diag_full['tile_name'] = f'{diag_elem[0]}_{diag_elem[1]}'
                diag_full['place'] = 'diag'

        #         right_full = pd.DataFrame(right['features'])
                #to_drop = fdf[fdf['place'] == 'right'][0].unique()
                to_drop = [el[1] for el in orig_right] if len(orig_right) > 0 else []
                right_full =  right_full.drop(right_full.index[list(set(to_drop))]).reset_index(drop = True)
                right_full['tile_name'] = f'{right_elem[0]}_{right_elem[1]}'
                right_full['place'] = 'right'

        #         down_full = pd.DataFrame(down['features'])
                #to_drop = fdf[fdf['place'] == 'down'][0].unique()
                to_drop = [el[1] for el in orig_down] if len(orig_down) > 0 else []
                down_full =  down_full.drop(down_full.index[list(set(to_drop))]).reset_index(drop = True)
                down_full['tile_name'] = f'{down_elem[0]}_{down_elem[1]}'
                down_full['place'] = 'down'

                ''' append all'''

                Block_df = orig_full.append(diag_full, ignore_index=True).append(right_full, ignore_index = True).append(down_full, ignore_index = True)
                Block_df['id'] = Block_df.id.astype(str)
                Block_df['geometry'] = Block_df.geometry.apply(lambda x : Polygon(x['coordinates'][0]))



                r_ = []
                if not diag_identify :
                    ''' remove diag '''
                    index_names_di = Block_df[ Block_df['place'] == 'diag' ].index.to_list()
                    r_.append(index_names_di)
                    # drop these row indexes from dataFrame
        #             Block_df = Block_df.drop(index_names).reset_index(drop = True)

                if not right_identify:
                    ''' remove right '''
                    index_names_r = Block_df[ Block_df['place'] == 'right' ].index.to_list()
                    r_.append(index_names_r)
                    # drop these row indexes from dataFrame


                if not down_identify:
                    ''' remove right '''
                    index_names_do = Block_df[ Block_df['place'] == 'down' ].index.to_list()
                    r_.append(index_names_do)
                    # drop these row indexes from dataFrame

                all_ind_remove = [item for sublist in r_ for item in sublist]

                Block_df = Block_df.drop(list(set(all_ind_remove))).reset_index(drop = True)
                Block_gdf = gpd.GeoDataFrame(Block_df[['geometry','id' ,'tile_name', 'place']], geometry = 'geometry').set_crs(epsg=4326, inplace=True)



                subtile_remov= to_rem[['geometry','tile_name', 'id']].drop_duplicates().reset_index(drop = True).dropna().reset_index(drop = True)
                subtile_remov['comb'] = subtile_remov['tile_name'] + "_" + subtile_remov['id']

                block_all_jsons = Block_gdf[['geometry','tile_name', 'id']].drop_duplicates().reset_index(drop = True).dropna().reset_index(drop = True)
                block_all_jsons['comb'] = block_all_jsons['tile_name'] + "_" + block_all_jsons['id']

#                 return block_all_jsons, subtile_remov, merged_
                ''' write to temp'''
                f_check = []
                if not block_all_jsons.empty :
                    f_check.append(block_all_jsons['geometry'].to_dict())
                    block_all_jsons.to_file(f'{fpo_name}/temp/block_{i}_{j}.geojson', driver='GeoJSON' )
                if not subtile_remov.empty:
                    f_check.append(subtile_remov['geometry'].to_dict())
                    subtile_remov.to_file(f'{fpo_name}/temp/subtileRemove_{i}_{j}.geojson', driver='GeoJSON' )
                if not merged_.empty:
                    f_check.append(merged_['geometry'].to_dict())
                    merged_[['geometry', 'place']].to_file(f'{fpo_name}/temp/merged_{i}_{j}.geojson', driver='GeoJSON' )
        except Exception as e :
            print(e, (i,j))

            
if __name__ == "__main__":
    
    global main_folder_path
    main_folder_path = sys.argv[1]
    
    ''' num of process'''


    cpus = multiprocessing.cpu_count()
    
    ''' make paths - temp/ , temp/result_f'''
    
    Path(f'{fpo_name}/temp').mkdir(parents=True, exist_ok=True)
    Path(f'{fpo_name}/temp/chak_with').mkdir(parents=True, exist_ok=True)
    Path(f'{fpo_name}/temp/result_f').mkdir(parents=True, exist_ok=True)
    
    c = glob.glob(f'{main_folder_path}/*.geojson')
    r = glob.glob(f'{main_folder_path}/*.geojson')

    c = list(set(list(map(lambda x : int(x.split('/')[-1].split('_')[1]), c))))
    r = list(set(list(map(lambda x : int(x.split('/')[-1].split('_')[-1].split('.')[0]), r))))

    print(len(c) ,len(r))
    

    
    parall_list = []
    for i in sorted(c):
        for j in sorted(r):
            parall_list.append((i,j))
    
    
    with multiprocessing.Pool(processes=cpus) as pool:
        pool.starmap(processor_subtile_remov, product(parall_list))
        

    folder = Path(fpo_name+"/temp/")
    shapefiles_block = folder.glob("block*")
    orig_full = pd.concat([
        gpd.read_file(shp)
        for shp in shapefiles_block
    ]).pipe(gpd.GeoDataFrame)


    shapefiles_block = folder.glob("subtile*")
    remove_full = pd.concat([
        gpd.read_file(shp)
        for shp in shapefiles_block
    ]).pipe(gpd.GeoDataFrame)

    merged_block = folder.glob("merged_*")
    merge_full = pd.concat([
        gpd.read_file(shp)
        for shp in merged_block
    ]).pipe(gpd.GeoDataFrame)
    
    
    
    o_r = orig_full[~orig_full['comb'].isin(remove_full['comb'])].drop_duplicates(subset='comb').reset_index(drop = True)
    
    final = o_r['geometry'].append(merge_full['geometry']).reset_index(drop = True)
    
    ff = gpd.GeoDataFrame(final)
    ff.rename(columns = {0 : 'geometry'}, inplace = True)
    ff['centroids'] = ff['geometry'].centroid
    
    valid_ff = ff[ff.geometry.is_valid].reset_index(drop = True)
    
    ''' declare fun here '''
    
    def return_overlap_(idx):
    
        ''' ip a point '''
        x_width ,y_width = 0.003401041030883789 ,0.0034069036154980026
        main_df = valid_ff.copy()
        x2,y2 = x_width/2,y_width/2
        x,y = main_df.iloc[idx]["centroids"].x,main_df.iloc[idx]["centroids"].y
        a,b = x-x2,y-y2
        chak = Polygon([[a,b],[a+x_width,b],[a+x_width,b+y_width],[a,b+y_width]])

        chak_with = gpd.GeoDataFrame(main_df[gpd.GeoDataFrame(main_df['geometry']).intersects(chak)], geometry='geometry')

        poly = main_df.loc[idx, 'geometry']
        intersected_index = []
        try:
            for ind, row in chak_with.iterrows():
                ''' shape_j is the anchor'''
                shape_i, shape_j = row['geometry'], poly
                if ind == idx:
                    continue
                if shape_i.intersection(shape_j).area/(shape_i.area + 1e-13) * 100 > 50 or shape_j.intersection(shape_i).area/(shape_j.area +  1e-13) * 100 > 50 :
                    if shape_i.area > shape_j.area :
                        intersected_index.append(idx)
                    else:
                        intersected_index.append(ind)
            intersected_index = list(set(intersected_index))

            if len(intersected_index) >0:
    #             print('overlap found')
                yess = np.array(intersected_index)


                ''' write to text file'''
                f_name = f"{fpo_name}/temp/chak_with/iter_{idx}.txt"
                with open(f_name, 'w') as f:
                    f.write(" ".join(map(str, yess)))

        except Exception as e :
            print(e, idx)
            
            
    mpp = valid_ff.index.values.tolist()
        
        
    with multiprocessing.Pool(processes=cpus) as pool:
        pool.starmap(return_overlap_, product(mpp))
        
    index_list= []
    for fx in glob.glob(fpo_name+"/temp/chak_with/*"):
        f = open(fx,"r")
        index_list.append(f.readlines()[0].split(" "))
        f.close()
    cvbto_rem = [int(item) for sublist in index_list for item in sublist]
    
    
    final_ppc = gpd.GeoDataFrame(valid_ff[~valid_ff.index.isin(list(set(cvbto_rem)))])
    
    areas = final_ppc.to_crs(epsg=3857).area.values
    
    final_ppc['area'] = np.round(areas * 0.0001, 3)
    final_ppc['centroid_x'] = final_ppc['geometry'].apply(lambda x: round(x.centroid.x, 3))
    final_ppc['centroid_y'] = final_ppc['geometry'].apply(lambda x: round(x.centroid.y, 3))
    final_ppc.drop('centroids', 1,inplace = True)
    
    final_ppc.to_file(fpo_name+'/'+fpo_name+'.geojson', driver='GeoJSON')
        
        
        
import geopandas as gpd
from itertools import combinations

def check_crs(lst):
    """
    Check whether the DataFrames within the list are of the same coordinate system.

    Parameters
    ----------
    lst : list[geopandas.GeoDataFrame]
        the list of GeoDataFrames to be checked

    Returns
    -------
    bool : True or False
        True implies that all the GeoDataFrames are of the same coordinate system, False otherwise.
    """
    for i, j in combinations(range(len(lst)), 2):
        if not lst[i].crs == lst[j].crs:
            print("The coordinate reference systems of the input DataFrames do not match. \
            Please transform the CRS before doing this function.")
            return False


def relation2dict(a_df, b_df, predicate="intersects"):
    """
    Detect the spatial relationship between the geometries in two GeoDataFrames
    and store the results in two dictionaries.

    Parameters
    ----------
    a_df : geopandas.GeoDataFrame
        the first GeoDataFrames
    b_df : geopandas.GeoDataFrame
        the second GeoDataFrames
    predicate : string (default 'intersects')
        Binary predicate. Valid values are determined by the spatial index used
        For details, please check the 'Predicates and Relationships' section of Shapely documentation

    Returns
    -------
    a_b_intersect_relation : dict
        The key is the index of geometries in 'a_df', the value is the index of geometries in 'b_df'
    b_a_intersect_relation : dict
        The key is the index of geometries in 'b_df', the value is the index of geometries in 'a_df'
    """
    # Check the CRS
    df_lst = [a_df, b_df]
    if check_crs(df_lst) == False:
        return

    # Create a spatial join
    merged = gpd.sjoin(a_df, b_df, predicate=predicate, how='inner')
    # Reduce the size of the DataFrame
    merged = merged[["index_right"]]
    # Group dataframe rows into list
    lst_a = merged.groupby(merged.index)['index_right'].apply(list)

    # Create an empty list by assigning the index as the key
    a_b_intersect_relation = {k: [] for k in a_df.index}
    b_a_intersect_relation = {k: [] for k in b_df.index}

    for idx in a_df.index:
        if idx in merged.index:
            a_b_intersect_relation[idx].extend(lst_a[idx])

    # Reverse the index in the reduced merged DataFrame
    merged = merged.reset_index().set_index("index_right", drop=True)
    # Group dataframe rows into list
    lst_b = merged.groupby(merged.index)['index'].apply(list)

    for idx in b_df.index:
        if idx in merged.index:
            b_a_intersect_relation[idx].extend(lst_b[idx])

    return a_b_intersect_relation, b_a_intersect_relation


def filter_block(block, building, predicate="intersects", num=0):

    block.reset_index(drop=True, inplace=True)
    building.reset_index(drop=True, inplace=True)

    # Get the relationship into two dicts
    block_relation, building_relation = relation2dict(block, building, predicate=predicate)

    # Remove empty blocks
    empty_lst = []
    for key, value in block_relation.items():
        if len(value) <= num:
            empty_lst.append(key)
    block = block.drop(empty_lst).reset_index(drop=True)

    return block


def filter_building(building, df_lst, predicate="intersects"):

    building.reset_index(drop=True, inplace=True)

    for df in df_lst:
        df.reset_index(drop=True, inplace=True)

    # Remove unrelated buildings
    related_lst = []
    for df in df_lst:
        su_relation, building_relation = relation2dict(df, building, predicate=predicate)
        for key, value in building_relation.items():
            if len(value) != 0:
                related_lst.append(key)

    # Remove duplicated index
    related_lst = list(set(related_lst))

    full_lst = list(building.index)
    remove_lst = [i for i in full_lst if i not in related_lst]

    building = building.drop(remove_lst).reset_index(drop=True)

    return building

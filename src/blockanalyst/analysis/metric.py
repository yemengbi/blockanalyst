import geopandas as gpd
import gemgis as gg
import numpy as np
import copy

from sklearn.metrics.pairwise import cosine_similarity
from varname import argname
from shapely.ops import transform, nearest_points
from shapely.ops import unary_union
from shapely.strtree import STRtree
from shapely.geometry import Point, LineString, LinearRing, Polygon

from utilities import relation2dict, check_crs
from tessellation import morphological_tessellation, voronoi_tessellation, slice_tessellation
from geometry import clean_geometry, filter_geometry, poly_of_best_fit, extract_points_from_poly

def internal_similarity(region, element, predicate="intersects"):
    """
    Calculate the cosine similarity of the elements within each spatial unit.

    Parameters
    ----------
    region : geopandas.GeoDataFrame
        the GeoDataFrame containing the spatial unit
    element : geopandas.GeoDataFrame
        the GeoDataFrame containing the elements for similarity analysis
    predicate : string (default 'intersects')
        Binary predicate. Valid values are determined by the spatial index used
        For details, please check the 'Predicates and Relationships' section of Shapely documentation

    Returns
    -------
    region_c : geopandas.GeoDataFrame
        the GeoDataFrame with the cosine similarity stored in a new column
    """
    # Specify the columns for calculation (exclude the geometry column)
    column = [x for x in element.columns if x!='geometry']
    # Specify the column to store data
    region_c = copy.deepcopy(region)
    region_c["internal_sim"] = ""

    region_relation, element_relation = relation2dict(region, element, predicate)

    for key, value in region_relation.items():
        if len(value) >= 2:
            similarities = cosine_similarity(element.loc[value, column].values)
            region_c.loc[key, "internal_sim"] = (np.sum(similarities) - len(value)) / (len(value) * (len(value)-1))

    return region_c


def area(gdf):
    """
    Calculate the area of input geometries.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        the GeoDataFrame containing the geometries

    Returns
    -------
    gdf : geopandas.GeoDataFrame
        the GeoDataFrame with an additional column for the area of geometries
    """
    gdf['area'] = ''
    for idx in gdf.index:
        gdf.loc[idx, "area"] = gdf.loc[idx, "geometry"].area

    return gdf


def area_density(region, element, trim=False):
    """
    Calculate the density of footprints of the elements within each spatial unit.

    Parameters
    ----------
    region : geopandas.GeoDataFrame
        the GeoDataFrame containing the spatial unit
    element : geopandas.GeoDataFrame
        the GeoDataFrame containing the elements for density calculation
    trim : bool (default False)
        True implies that the footprint of each element will be trimmed according to
        the boundary of spatial unit, False otherwise.

    Returns
    -------
    region_c : geopandas.GeoDataFrame
        the GeoDataFrame with the density calculation result stored in a new column
    """
    # Get the name of the arguments
    name = argname('element')
    name = name + '_area_dens'

    region_c = copy.deepcopy(region)
    region_c[name] = 0

    # ----- Calculate area -----
    region_c = area(region_c)
    element = area(element)

    # ----- Calculate density -----
    # Calculate the relationships between the two DataFrames of polygons
    a_b_intersect_relation, _ = relation2dict(region, element)

    for key, value in a_b_intersect_relation.items():
        if not len(value) == 0:
            element_area_lst = []
            if not trim:
                element_area_lst = [element.loc[i, 'area'] for i in value]
                # Calculate the density
                region_c.loc[key, name] = sum(element_area_lst) / region_c.loc[key, 'area']
            else:
                for i in value:
                    # Get the intersection between two polygons
                    intersect = element.geometry[i].intersection(region.geometry[key])
                    # Explode the result if it's a GeometryCollection
                    if intersect.geom_type == 'GeometryCollection':
                        exploded = gg.vector.explode_geometry_collection(collection=intersect)
                        int_area = 0
                        for geom in exploded:
                            int_area = int_area + geom.area
                    else:
                        # if not, just directly calculate the area of the intersection
                        int_area = intersect.area
                    element_area_lst.append(int_area)
                # Calculate the density
                region_c.loc[key, name] = sum(element_area_lst) / region_c.loc[key, 'area']

    return region_c


def num_density(region, element, predicate="intersects"):
    """
    Calculate the number of elements and the corresponding density within each spatial unit.

    Parameters
    ----------
    region : geopandas.GeoDataFrame
        the GeoDataFrame containing the spatial unit
    element : geopandas.GeoDataFrame
        the GeoDataFrame containing the elements for density calculation
    predicate : string (default 'intersects')
        Binary predicate. Valid values are determined by the spatial index used
        For details, please check the 'Predicates and Relationships' section of Shapely documentation

    Returns
    -------
    region_c : geopandas.GeoDataFrame
        the GeoDataFrame with the density calculation result stored in a new column
    """
    # Get the name of the arguments
    name = argname('element')
    name_n = name + '_num'
    name_dens = name + '_num_dens'

    region_c = copy.deepcopy(region)
    region_c[name_n] = 0
    region_c[name_dens] = 0

    # ----- Calculate area -----
    region_c = area(region_c)
    element = area(element)

    # ----- Calculate density -----
    a_b_intersect_relation, _ = relation2dict(region, element, predicate)

    for key, value in a_b_intersect_relation.items():
        region_c.loc[key, name_n] = len(value)
        region_c.loc[key, name_dens] = len(value) / region_c.loc[key, 'area']

    return region_c


def setback(region, element, predicate="intersects"):
    """
    Calculate the setbacks from each element to the corresponding spatial unit.

    Parameters
    ----------
    region : geopandas.GeoDataFrame
        the GeoDataFrame containing the spatial unit
    element : geopandas.GeoDataFrame
        the GeoDataFrame containing the elements for setback calculation
    predicate : string (default 'intersects')
        Binary predicate. Valid values are determined by the spatial index used
        For details, please check the 'Predicates and Relationships' section of Shapely documentation

    Returns
    -------
    region_c : geopandas.GeoDataFrame
        the GeoDataFrame with the setback calculation result stored in a new column
    """
    name = argname('element')
    name_avg = name + '_avg_setback'

    region_c = copy.deepcopy(region)
    a_b_intersect_relation, _ = relation2dict(region, element, predicate)

    # ----- Calculate the setbacks -----
    setbacks = {k: {} for k in range(0, len(region))}

    for key, value in a_b_intersect_relation.items():
        if not len(value) == 0:
            # Extract the vertices coordinates of the spatial units
            region_vertices = region.geometry[key].boundary.coords
            # Create edge (LineString) based on these coordinates
            region_edge = [LineString(region_vertices[k:k+2]) for k in range(len(region_vertices)-1)]
            region_edge_series = gpd.GeoSeries(region_edge)

            distance = {}
            for idx in value:
                # Calculate distance from each edge to the element
                dist = region_edge_series.distance(element.geometry[idx])
                # The key is the index of element; the value is the minimum distance to all the edges
                distance[idx] = min(dist)

            # The key is the index of spatial unit; the value is the dict of distance
            setbacks[key] = distance

    # ----- Calculate the average setbacks -----
    avg_setback = {}

    for key, value in setbacks.items():
        if not len(value) == 0:
            setback_value = list(value.values())
            avg_setback[key] = sum(setback_value) / len(value)

    region_c[name_avg] = ''
    # Match the DataFrame with the dict
    for key, value in avg_setback.items():
        region_c.loc[key, name_avg] = value

    return region_c


def cul_de_sac(region, street, predicate="contains"):
    """
    Calculate the number and total length of cul-de-sacs within the spatial units.

    Parameters
    ----------
    region : geopandas.GeoDataFrame
        the GeoDataFrame containing the spatial unit
    street : geopandas.GeoDataFrame
        the GeoDataFrame containing all the streets
    predicate : string (default 'contains')
        Binary predicate. Valid values are determined by the spatial index used
        For details, please check the 'Predicates and Relationships' section of Shapely documentation

    Returns
    -------
    region_c : geopandas.GeoDataFrame
        the GeoDataFrame with two new columns with the data of cul-de-sacs
    """
    region_c = copy.deepcopy(region)
    region_c["cul_len"] = 0

    a_b_intersect_relation, _ = relation2dict(region, street, predicate)

    for key, value in a_b_intersect_relation.items():
        if not len(value) == 0:
            cul_len = []
            for idx in value:
                cul_len.append(street.geometry[idx].length)
            region_c.loc[key, "cul_len"] = sum(cul_len)
            region_c.loc[key, "cul_num"] = len(cul_len)

    # The column of 'cul_num' may contain NAN
    region_c = region_c.fillna(0)

    return region_c


def eccentricity(region, element, predicate="intersects"):
    """
    Calculate the average distance between each element and the center of the spatial unit.

    Parameters
    ----------
    region : geopandas.GeoDataFrame
        the GeoDataFrame containing the spatial unit
    element : geopandas.GeoDataFrame
        the GeoDataFrame containing the elements for eccentricity calculation
    predicate : string (default 'intersects')
        Binary predicate. Valid values are determined by the spatial index used
        For details, please check the 'Predicates and Relationships' section of Shapely documentation

    Returns
    -------
    region_c : geopandas.GeoDataFrame
        the GeoDataFrame with the eccentricity calculation result stored in a new column
    """
    region_c = copy.deepcopy(region)

    # Get the name of the arguments
    name = argname('element')
    name = name + '_eccentricity'

    # Calculate the relationships between the two DataFrames of polygons
    a_b_intersect_relation, _ = relation2dict(region, element, predicate)

    for key, value in a_b_intersect_relation.items():
        if not len(value) == 0:
            element_dist_lst = []
            for idx in value:
                element_dist_lst.append(region.geometry[key].centroid.distance(element.geometry[idx].centroid))
            region_c.loc[key, name] = sum(element_dist_lst) / len(value)

    return region_c


def division_degree(region, element, predicate="contains", tessellation='voronoi'):
    """
    Calculate the division degree of each spatial unit by the internal elements.

    Parameters
    ----------
    region : geopandas.GeoDataFrame
        the GeoDataFrame containing the spatial unit
    element : geopandas.GeoDataFrame
        the GeoDataFrame containing the elements for division degree calculation
    predicate : string (default 'intersects')
        Binary predicate. Valid values are determined by the spatial index used
        For details, please check the 'Predicates and Relationships' section of Shapely documentation
    tessellation : string (default 'voronoi')
        The calculation of division can be based on two approaches, namely the 'voronoi' or 'morphological'.
        The morphological tessellation takes more time, but also leading to more accurate result.

    Returns
    -------
    region_c : geopandas.GeoDataFrame
        the GeoDataFrame with the eccentricity calculation result stored in a new column
    """
    region_c = copy.deepcopy(region)
    region_c = area(region_c)
    element = clean_geometry(element)

    name = argname('element')
    name = name + '_division'

    def power(num):
        powered = num ** 2
        return powered

    if tessellation == 'voronoi':
        tess = voronoi_tessellation(element)
        # Slice the tessellation result by the boundary of spatial unit
        parcels = slice_tessellation(tess, region, element)
    elif tessellation == 'morphological':
        # The morphological tessellation function does not accept Point or Linestring
        element = filter_geometry(element, types=['Polygon'])
        tess = morphological_tessellation(element)
        # Slice the tessellation result by the boundary of spatial unit
        parcels = slice_tessellation(tess, region, element)
    else:
        return

    parcels = area(parcels)
    # Calculate the relationships between the two DataFrames of polygons
    a_b_intersect_relation, _ = relation2dict(region, parcels, predicate)

    for key, value in a_b_intersect_relation.items():
        if not len(value) == 0:
            # Get the areas of different parcels in a list
            parcel_area_lst = list(parcels.loc[value]['area'])
            # Calculate the area of open spaces within each spatial unit
            open_space_area = region_c.geometry[key].area - sum(parcel_area_lst)
            parcel_area_lst.append(open_space_area)

            # Calculate the degree of division
            ratio = [x / region_c.geometry[key].area for x in parcel_area_lst]
            x_power = list(map(power, ratio))
            x_degree = sum(x_power)
            region_c.loc[key, name] = 1 - x_degree

    return region_c


def avg_distance_btw_polys(target, other, threshold=0.5, distance=30):
    """
    Calculate the average distance between the boundaries of the 'target' polygons and their polygon of
    best fit (PBF).

    Parameters
    ----------
    target : geopandas.GeoDataFrame
        the GeoDataFrame of spatial units to be fitted
    other : geopandas.GeoDataFrame
        the GeoDataFrame of another group of spatial units that can be combined to align
        with the 'target' division
    threshold : float (default 0.5)
        only the geometries in 'other' whose overlapping area with the corresponding 'target' geometry is
        greater than the threshold will be included as a part of the PBF
    distance : float
        the distance between every two consecutive sample points when extracting points from the 'target' geometry

    Returns
    -------
    target : geopandas.GeoDataFrame
        the GeoDataFrame with two additional columns indicating the average distance to its PBF, as well as the
        standard deviation of the distances between the sample points and the PBF
    """
    target = clean_geometry(target)
    other = clean_geometry(other)

    fitting = poly_of_best_fit(target, other, threshold)

    target["avg_dist_btw_polys"] = ""
    target["std_dist_btw_polys"] = ""

    for i in target.index:
        if not fitting[i] == []:
            pts = extract_points_from_poly(target.geometry[i], distance)
            output_pts = []
            for geom in pts:
                output_pts.append(geom.distance(fitting[i].boundary))
            target.loc[i, "avg_dist_btw_polys"] = sum(output_pts) / len(output_pts)
            target.loc[i, "std_dist_btw_polys"] = np.std(output_pts)

    return target


def intersection_over_union(target, other, threshold=0.5):
    """
    Calculate the intersection over union of the 'target' polygons and their polygon of best fit (PBF).

    Parameters
    ----------
    target : geopandas.GeoDataFrame
        the GeoDataFrame of spatial units to be fitted
    other : geopandas.GeoDataFrame
        the GeoDataFrame of another group of spatial units that can be combined to align
        with the 'target' division
    threshold : float (default 0.5)
        only the geometries in 'other' whose overlapping area with the corresponding 'target' geometry is
        greater than the threshold will be included as a part of the PBF

    Returns
    -------
    target : geopandas.GeoDataFrame
        the GeoDataFrame with an additional column indicating the intersection over union of the 'target' geometries
        and their PBF
    """
    target = clean_geometry(target)
    other = clean_geometry(other)

    fitting = poly_of_best_fit(target, other, threshold)
    target["IoU"] = ""

    for i in target.index:
        if not fitting[i] == []:
            ## Intersection
            intersect_area = []
            # Calculate the intersection between target geometry and its PBFC
            intersection = target.geometry[i].intersection(fitting[i])
            # The intersection in some cases would be GeometryCollection
            if intersection.geom_type == "GeometryCollection":
                GeometryCollection_exploded = gg.vector.explode_geometry_collection(collection=intersection)
                for geom in GeometryCollection_exploded:
                    # If the intersection is just a polygon, it would be easy to calculate
                    if geom.geom_type == "Polygon" or geom.geom_type == "MultiPolygon":
                        intersect_area.append(geom.area)
            else:
                if intersection.geom_type == "Polygon" or intersection.geom_type == "MultiPolygon":
                    intersect_area.append(intersection.area)
            ## Union
            union_area = target.geometry[i].union(fitting[i]).area

            target.loc[i, "IoU"] = sum(intersect_area) / union_area

    return target

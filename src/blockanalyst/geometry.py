import pandas as pd
import geopandas as gpd
import pygeos
import numpy as np
import gemgis as gg
import copy

from shapely.geometry import Polygon, LinearRing, LineString
from shapely.strtree import STRtree
from shapely.ops import nearest_points

def clean_geometry(gdf):
    """
    Conduct pre-processing for the given dataset,
    including removing empty geometries and explode multipart geometries.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        the GeoDataFrame to be cleaned

    Returns
    -------
    gdf_cleaned : geopandas.GeoDataFrame
        the cleaned version of the GeoDataFrame
    """
    # Remove empty geometry
    gdf_filtered = gdf[~gdf.is_empty]
    # Explode multipart geometries into multiple single geometries
    gdf_exploded = gdf_filtered.explode(ignore_index=True)
    # Reset the index
    gdf_cleaned = gdf_exploded.reset_index(drop=True)

    return gdf_cleaned


def filter_geometry(gdf, types=None):
    """
    Keep only the selected types of geometries.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        the GeoDataFrame to be filtered

    Returns
    -------
    gdf_geom_types : geopandas.GeoDataFrame
        the GeoDataFrame with only the selected type of geometries
    """
    if types is None:
        types = ["Polygon"]

    # Select specific types of geometries
    gdf_geom_types = gdf[gdf.geom_type.isin(types)]

    return gdf_geom_types


def explode_collection(gdf):
    """
    Explode all the Shapely Geometry Collections within the DataFrame to Base Geometries.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        the GeoDataFrame to be processed

    Returns
    -------
    gdf_exploded : geopandas.GeoDataFrame
        the GeoDataFrame with all the Geometry Collections been exploded
    """
    gdf_geometrycollection = gdf[gdf.geom_type == 'GeometryCollection']
    collection = []

    for geom in gdf_geometrycollection.geometry:
        geometrycollection_exploded = gg.vector.explode_geometry_collection(collection=geom)
        collection.extend(geometrycollection_exploded)

    collection_gdf = gpd.GeoDataFrame(geometry=collection, crs=gdf.crs)
    gdf_not_geometrycollection = gdf[gdf.geom_type != 'GeometryCollection']

    gdf_exploded = pd.concat([gdf_not_geometrycollection.geometry, collection_gdf.geometry])

    return gdf_exploded


def dissolve_poly(gdf):
    """
    Dissolve the geometries with the relationship of either touch, intersect, overlap or contain
    within a given GeoDataFrame.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        the GeoDataFrame which the geometries belong to

    Returns
    -------
    gdf_dissolved : geopandas.GeoDataFrame
        the GeoDataFrame with all the joint geometries been dissolved
    """
    # Data Pre-processing
    gdf_cleaned = clean_geometry(gdf)
    # Assign a new column for overlapping information
    gdf_cleaned = gdf_cleaned.assign(overlap=np.nan).reset_index(drop=True)

    # Create a dict to act as an adjacency list
    key_list = list(range(0, len(gdf_cleaned)))
    overlaps = {k: [] for k in key_list}

    # Create a dict to match geometries with the index in the DataFrame
    # Since Shapely geometry is not hashable, here we use 'PyGEOS'
    index_by_geom = {}

    for i in range(len(gdf_cleaned)):
        index_by_geom[pygeos.from_shapely(gdf_cleaned.geometry[i])] = i

    # Create a list of PyGEOS geometries
    pg_list = []

    for i in range(len(gdf_cleaned)):
        pg_list.append(pygeos.from_shapely(gdf_cleaned.geometry[i]))

    # Get the index of all the joint geometries into the dict of 'overlaps'
    # Create a STRtree
    tree = pygeos.STRtree(pg_list)

    for i in range(len(pg_list)):
        query = tree.query(pg_list[i])
        for j in range(len(query)):
            if not (pg_list[i] == pg_list[query[j]]):
                if not pygeos.disjoint(pg_list[i], pg_list[query[j]]):
                    overlaps[index_by_geom[pg_list[i]]].append(query[j])

    # Use the Depth First Search Algorithm
    # Set to keep track of visited nodes of graph
    visited = set()

    def DFS(visited, graph, lst, node):
        if node not in visited:
            lst.append(node)
            visited.add(node)
            for neighbour in graph[node]:
                DFS(visited, graph, lst, neighbour)

        return lst

    visited_keys = set()
    overlaps_clusters = []

    for node in overlaps.keys():
        if node not in visited_keys:
            lst = []
            lst = DFS(visited, overlaps, lst, node)
            overlaps_clusters.append(lst)
            visited_keys.add(node)

    # Assign each cluster with a unique number
    n = 1
    for lst in overlaps_clusters:
        if len(lst) != 1 and len(lst) != 0:
            for i in lst:
                gdf_cleaned.loc[i, "overlap"] = n
            n = n + 1

    # Use the df.dissolve to merge those geometries with the same label
    # Slice Pandas DataFrame by row
    gdf_na = gdf_cleaned[gdf_cleaned.overlap.isna()]
    gdf_not_na = gdf_cleaned[gdf_cleaned.overlap.notna()]

    # Dissolve the overlapping geometry
    gdf_dissolved_not_na = gdf_not_na.dissolve(by='overlap')
    gdf_dissolved = gpd.GeoDataFrame(
        pd.concat([gdf_na, gdf_dissolved_not_na], ignore_index=True), crs=gdf.crs)
    gdf_dissolved.reset_index(drop=True, inplace=True)

    return gdf_dissolved


def poly2line(gdf):
    """
    Decompose the Polygons into LineStrings within a given DataFrame.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        the GeoDataFrame which the geometries belong to

    Returns
    -------
    gdf_decomposed : geopandas.GeoDataFrame
        the GeoDataFrame with all the Polygons been decomposed to LineStrings
    """
    # Explode multi-part geometries
    gdf_cleaned = clean_geometry(gdf)

    # Select the Polygon geometries within the DataFrame
    gdf_poly = gdf_cleaned[gdf_cleaned.geom_type == 'Polygon']
    # Select the Linestring geometries within the DataFrame
    gdf_line = gdf_cleaned[gdf_cleaned.geom_type == 'LineString']

    segments = []

    for poly in gdf_poly.geometry:
        boundary = poly.boundary
        if boundary.type == 'MultiLineString':
            for line in boundary.geoms:
                segments.append(line)
        else:
            segments.append(boundary)

    gdf_segment = gpd.GeoDataFrame(geometry=segments, crs=gdf)
    gdf_decomposed = pd.concat([gdf_line.geometry, gdf_segment.geometry])

    return gdf_decomposed


def close_holes(gdf):
    """
    Close polygon holes within a DataFrame.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        the GeoDataFrame which the geometries belong to

    Returns
    -------
    gdf_filled : geopandas.GeoDataFrame
        the GeoDataFrame with all the polygon holes been filled
    """
    # Close polygon holes
    for i in gdf.index:
        if not gdf.geometry[i].geom_type != 'Polygon':
            if gdf.geometry[i].interiors:
                gdf.geometry[i] = Polygon(list(gdf.geometry[i].exterior.coords))

    # Remove those polygons that are enclosed by the others
    tree_poly = STRtree(gdf.geometry)
    holes = []

    for i in gdf.index:
        # Get the query data (a list) for street intersection
        query = tree_poly.query(gdf.geometry[i])
        # Check the relationship
        for x in query:
            if not x.almost_equals(gdf.geometry[i]):
                if x.contains(gdf.geometry[i]):
                    holes.append(i)

    gdf_filled = gdf.drop(holes)

    return gdf_filled


def filter_by_overlaps(target, other, threshold=0.95):
    """
    Filter out the geometries from 'target' that have an overlap with 'other' of a ratio smaller than the
    designated threshold.
    This function helps to identify the spatial units that are mostly covered by another spatial division.
    Notes: This function assumes that 'other' is a fully partitioning of space and can be dissolved into
    a single entity easily.

    Parameters
    ----------
    target : geopandas.GeoDataFrame
        the GeoDataFrame to be filtered
    other : geopandas.GeoDataFrame
        the GeoDataFrame containing the polygons to be overlapped with
    threshold : float (default 0.95)

    Returns
    -------
    target_filtered : geopandas.GeoDataFrame
        the already filtered 'target' GeoDataFrame
    """
    # Dissolve the polygons of 'other'
    other["dissolve"] = 1
    other_diss = other.dissolve(by='dissolve')

    removal = []

    for i in target.index:
        overlap_area = []
        # Not sure if the dissolved 'other' would contain multiple geometries
        for j in other_diss.index:
            # Calculate the intersection between dissolved 'other' and the 'target'
            intersection = other_diss.geometry[j].intersection(target.geometry[i])
            # The intersection in some cases would be GeometryCollection
            if intersection.geom_type == "GeometryCollection":
                GeometryCollection_exploded = gg.vector.explode_geometry_collection(collection=intersection)
                for geom in GeometryCollection_exploded:
                    # If the intersection is just a polygon, it would be easy to calculate
                    if geom.geom_type == "Polygon":
                        overlap_area.append(geom.area)
            else:
                if intersection.geom_type == "Polygon" or intersection.geom_type == "MultiPolygon":
                    overlap_area.append(intersection.area)
        overlap_ratio = sum(overlap_area) / target.geometry[i].area

        # Keep only those spatial units that have the ratio over the threshold
        if not overlap_ratio >= threshold:
            removal.append(i)

    if len(removal) != 0:
        target_copy = copy.deepcopy(target)
        target_filtered = target_copy.drop(removal)
    else:
        target_filtered = target

    return target_filtered


def fit_poly(target, other, threshold=0.5):
    """
    Keep only those spatial units in 'other' whose overlapping area with the corresponding 'target' geometry
    is greater than the threshold.
    This will serve as the prerequisite for obtaining the polygon of best fit (PBF).

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
    fitting : geopandas.GeoDataFrame
        the GeoDataFrame containing the resulting geometries, which is obtained through combining geometries
        from 'other' in order to align with the geometries in 'target'
    """
    fitting = {k: [] for k in target.index}
    tree = STRtree(other.geometry)

    for i in target.index:
        query = tree.query(target.geometry[i])
        # Calculate the geometry of intersection and the ratio of overlapping
        for x in query:
            intersection = x.intersection(target.geometry[i])
            # The intersection in some cases would be GeometryCollection
            if intersection.geom_type == "GeometryCollection":
                GeometryCollection_exploded = gg.vector.explode_geometry_collection(collection=intersection)
                area = []
                for geom in GeometryCollection_exploded:
                    # If the intersection is just a polygon, it would be easy to calculate
                    if geom.geom_type == "Polygon":
                        area.append(geom.area)
                overlap_ratio_other = sum(area) / x.area
                overlap_ratio_target = sum(area) / target.geometry[i].area
            else:
                if intersection.geom_type == "Polygon" or intersection.geom_type == "MultiPolygon":
                    overlap_ratio_other = intersection.area / x.area
                    overlap_ratio_target = intersection.area / target.geometry[i].area
                else:
                    overlap_ratio_other = 0
                    overlap_ratio_target = 0

            # Keep only those spatial units that have the ratio over 50%
            if overlap_ratio_other >= threshold or overlap_ratio_target >= threshold:
                fitting[i].append(x)

    return fitting


def poly_of_best_fit(target, other, threshold=0.5):
    """
    The polygon of best fit (PBF) is a geometric construct that optimally represents the amalgamation
    of spatial units intersected with the 'target' division.
    It is formed by combining contiguous spatial units in 'other' that share a significant portion of their area
    with the corresponding 'target' unit.

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
    fitting : geopandas.GeoDataFrame
        the GeoDataFrame containing the resulting geometries, which is obtained through combining geometries
        from 'other' in order to align with the geometries in 'target'
    """
    target = clean_geometry(target)
    other = clean_geometry(other)

    fitting = fit_poly(target, other, threshold)

    def linking(df):
        dissolved = dissolve_poly(df)

        # There may be some polygons that are not connected
        # We try to connect these polygons through a LineString, and convert it to a polygon
        geom_lst = list(dissolved.geometry)

        connector = []

        if len(geom_lst) > 1:
            for i in range(len(geom_lst)-1):
                # Create a tree by slicing the list to the elements after the index
                tree = STRtree(geom_lst[i + 1:])
                # Query for the nearest polygon
                query = tree.nearest(geom_lst[i])
                # Get the nearest points of these two polygons
                nearest_pts = nearest_points(geom_lst[i], query)
                # Create a LineString connecting the two nearest points
                line = LineString([nearest_pts[0], nearest_pts[1]])
                # Generate a polygon by applying a buffer around the line
                line_poly = Polygon(line.buffer(0.0001).exterior)
                # Add the polygon to the list
                connector.append(line_poly)

        geom_lst.extend(connector)
        new_df = gpd.GeoDataFrame(geometry=geom_lst, crs=df.crs)
        new_dissolved = dissolve_poly(new_df)

        return new_dissolved

    for key, value in fitting.items():
        if len(value) > 1:
            gdf = gpd.GeoDataFrame(geometry=value, crs=target.crs)
            # Loop until there is only one polygon
            while len(gdf) > 1:
                gdf = linking(gdf)
            fitting[key] = gdf.geometry[0]
        else:
            if len(value) == 1:
                fitting[key] = value[0]

    return fitting


def extract_points_from_poly(polygon, distance):
    """
    Extract points from the boundary of a polygon at equal distances.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        the polygon for point extraction
    distance : float
        the distance between every two consecutive points

    Returns
    -------
    points : list[shapely.geometry.Point]
        the resulting points from the boundary of the polygon
    """
    # Create a linear ring from the polygon exterior
    exterior = LinearRing(polygon.exterior.coords)
    # Convert the linear ring to a line string
    line = LineString(exterior)

    # Extract points from the line string at equal distances
    points = [line.interpolate(d) for d in np.arange(0, line.length, distance)]

    return points

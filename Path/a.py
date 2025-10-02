import pandas as pd
import numpy as np
import math
from typing import List, Dict, Tuple, Optional, Set
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
import heapq
from dataclasses import dataclass
from collections import defaultdict
import warnings
import os
warnings.filterwarnings('ignore')

# Try to import OR-Tools for TSP (optional)
try:
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp
    HAS_ORTOOLS = True
except ImportError:
    HAS_ORTOOLS = False
    print("OR-Tools not available. Will use   pathfinding for all routing.")

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate Haversine distance between two points in kilometers."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    return 6371.0 * 2 * math.asin(math.sqrt(a))

class AStarPathfinder:
    """
    A* pathfinding implementation for POI routing using Haversine distance.
    Optimized for geographical coordinates.
    """
    
    def __init__(self, nodes: Dict[str, Tuple[float, float]]):
        """
        Initialize pathfinder with node coordinates.
        
        Parameters:
        nodes: Dictionary mapping node_id -> (latitude, longitude)
        """
        self.nodes = nodes
        self._distance_cache = {}  # Cache for distance calculations
    
    def _get_distance(self, node1_id: str, node2_id: str) -> float:
        """Get cached distance between two nodes."""
        cache_key = tuple(sorted([node1_id, node2_id]))
        if cache_key not in self._distance_cache:
            lat1, lon1 = self.nodes[node1_id]
            lat2, lon2 = self.nodes[node2_id]
            self._distance_cache[cache_key] = haversine_distance(lat1, lon1, lat2, lon2)
        return self._distance_cache[cache_key]
    
    def _heuristic(self, node_id: str, goal_id: str) -> float:
        """
        Heuristic function for A*. Uses Haversine distance as admissible heuristic.
        """
        return self._get_distance(node_id, goal_id)
    
    def _get_neighbors(self, node_id: str, valid_nodes: Set[str]) -> List[str]:
        """Get all valid neighboring nodes."""
        return [neighbor for neighbor in valid_nodes if neighbor != node_id]
    
    def _reconstruct_path(self, came_from: Dict[str, str], current: str) -> List[str]:
        """Reconstruct path from came_from dictionary."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]  # Reverse to get start->goal order
    
    def find_shortest_path(self, start_id: str, goal_id: str, 
                          valid_nodes: Optional[Set[str]] = None) -> Tuple[List[str], float]:
        """
        Find shortest path between two nodes using A* algorithm.
        
        Parameters:
        start_id: Starting node ID
        goal_id: Goal node ID
        valid_nodes: Optional set of valid node IDs (if None, uses all nodes)
        
        Returns:
        Tuple of (path_as_list_of_ids, total_distance_km)
        """
        if valid_nodes is None:
            valid_nodes = set(self.nodes.keys())
        
        if start_id not in valid_nodes or goal_id not in valid_nodes:
            raise ValueError("Start or goal node not in valid nodes set")
        
        if start_id == goal_id:
            return [start_id], 0.0
        
        # A* algorithm implementation
        open_set = [(0.0, start_id)]  # Priority queue: (f_score, node_id)
        came_from = {}  # node_id -> parent_node_id
        g_score = {start_id: 0.0}  # Actual cost from start
        f_score = {start_id: self._heuristic(start_id, goal_id)}  # g + h
        
        open_set_hash = {start_id}  # For efficient membership testing
        
        while open_set:
            current_f, current = heapq.heappop(open_set)
            open_set_hash.remove(current)
            
            if current == goal_id:
                path = self._reconstruct_path(came_from, current)
                return path, g_score[current]
            
            for neighbor in self._get_neighbors(current, valid_nodes):
                tentative_g = g_score[current] + self._get_distance(current, neighbor)
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    # This path to neighbor is better than any previous one
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal_id)
                    
                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)
        
        # No path found
        raise ValueError(f"No path found from {start_id} to {goal_id}")
    
    def find_optimal_tour(self, node_list: List[str], 
                         start_node: Optional[str] = None) -> Tuple[List[str], float]:
        """
        Find optimal tour visiting all nodes using A* for pathfinding.
        Uses nearest neighbor heuristic with A* paths between consecutive nodes.
        
        Parameters:
        node_list: List of node IDs to visit
        start_node: Optional starting node (if None, uses first node)
        
        Returns:
        Tuple of (ordered_node_list, total_distance)
        """
        if len(node_list) <= 1:
            return node_list, 0.0
        
        if len(node_list) == 2:
            path, distance = self.find_shortest_path(node_list[0], node_list[1])
            return path, distance
        
        valid_nodes = set(node_list)
        
        if start_node is None:
            start_node = node_list[0]
        
        if start_node not in valid_nodes:
            raise ValueError("Start node not in node list")
        
        # Use nearest neighbor heuristic to determine visit order
        unvisited = set(node_list)
        current = start_node
        tour_order = [current]
        unvisited.remove(current)
        
        # Build tour order using nearest neighbor
        while unvisited:
            nearest = min(unvisited, key=lambda x: self._get_distance(current, x))
            tour_order.append(nearest)
            current = nearest
            unvisited.remove(current)
        
        # Now find A* paths between consecutive nodes in the tour
        complete_path = []
        total_distance = 0.0
        
        for i in range(len(tour_order) - 1):
            start = tour_order[i]
            end = tour_order[i + 1]
            
            # Find A* path between consecutive tour nodes
            segment_path, segment_distance = self.find_shortest_path(start, end, valid_nodes)
            
            # Add to complete path (avoid duplicating nodes)
            if i == 0:
                complete_path.extend(segment_path)
            else:
                complete_path.extend(segment_path[1:])  # Skip first node (duplicate)
            
            total_distance += segment_distance
        
        return complete_path, total_distance

def load_and_preprocess_data(csv_path: str, 
                           kabupaten_name: str,
                           categories_of_interest: Optional[List[str]] = None,
                           verbose: bool = True) -> pd.DataFrame:
    """
    Load and preprocess POI data from CSV file.
    
    Parameters:
    csv_path: Path to the CSV file
    kabupaten_name: Name of kabupaten to filter for
    categories_of_interest: Optional list of categories to filter for
    verbose: Whether to print preprocessing steps
    
    Returns:
    Preprocessed DataFrame ready for routing analysis
    """
    if verbose:
        print(f"Loading data from: {csv_path}")
    
    # Check if file exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Read CSV file
    try:
        df = pd.read_csv(csv_path)
        if verbose:
            print(f"✓ Loaded {len(df)} rows from CSV")
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")
    
    # Validate required columns
    required_columns = ['id', 'latitude', 'longitude', 'category', 'kabupaten_kota']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    if verbose:
        print(f"✓ All required columns present: {required_columns}")
    
    # Step 1: Drop rows with missing coordinates
    initial_count = len(df)
    df = df.dropna(subset=['latitude', 'longitude']).copy()
    dropped_coords = initial_count - len(df)
    
    if verbose and dropped_coords > 0:
        print(f"✓ Dropped {dropped_coords} rows with missing coordinates")
    elif verbose:
        print("✓ No rows with missing coordinates found")
    
    if len(df) == 0:
        raise ValueError("No valid POIs remaining after dropping missing coordinates")
    
    # Step 2: Filter for specific kabupaten
    if kabupaten_name not in df['kabupaten_kota'].values:
        available_kabupaten = df['kabupaten_kota'].unique()
        raise ValueError(f"Kabupaten '{kabupaten_name}' not found. Available: {list(available_kabupaten)}")
    
    df_filtered = df[df['kabupaten_kota'] == kabupaten_name].copy()
    if verbose:
        print(f"✓ Filtered to {kabupaten_name}: {len(df_filtered)} POIs")
    
    if len(df_filtered) == 0:
        raise ValueError(f"No POIs found for kabupaten: {kabupaten_name}")
    
    # Step 3: Filter by categories if specified
    if categories_of_interest is not None:
        available_categories = df_filtered['category'].unique()
        valid_categories = [cat for cat in categories_of_interest if cat in available_categories]
        invalid_categories = [cat for cat in categories_of_interest if cat not in available_categories]
        
        if invalid_categories and verbose:
            print(f"⚠ Warning: Categories not found: {invalid_categories}")
        
        if not valid_categories:
            raise ValueError(f"None of the specified categories found. Available: {list(available_categories)}")
        
        df_filtered = df_filtered[df_filtered['category'].isin(valid_categories)].copy()
        if verbose:
            print(f"✓ Filtered to categories {valid_categories}: {len(df_filtered)} POIs")
    
    # Step 4: Data type validation and conversion
    try:
        df_filtered['latitude'] = pd.to_numeric(df_filtered['latitude'], errors='coerce')
        df_filtered['longitude'] = pd.to_numeric(df_filtered['longitude'], errors='coerce')
        
        # Drop any rows where conversion failed
        df_filtered = df_filtered.dropna(subset=['latitude', 'longitude'])
        
        if verbose:
            print(f"✓ Validated coordinate data types: {len(df_filtered)} POIs remaining")
    except Exception as e:
        raise ValueError(f"Error converting coordinates to numeric: {e}")
    
    # Step 5: Basic data validation
    lat_range = df_filtered['latitude'].min(), df_filtered['latitude'].max()
    lon_range = df_filtered['longitude'].min(), df_filtered['longitude'].max()
    
    # Rough validation for Indonesia coordinates
    if not (-11 <= lat_range[0] <= 6 and -11 <= lat_range[1] <= 6):
        print(f"⚠ Warning: Latitude range {lat_range} seems unusual for Indonesia")
    if not (95 <= lon_range[0] <= 141 and 95 <= lon_range[1] <= 141):
        print(f"⚠ Warning: Longitude range {lon_range} seems unusual for Indonesia")
    
    # Reset index
    df_filtered = df_filtered.reset_index(drop=True)
    
    if verbose:
        print(f"\n=== PREPROCESSING SUMMARY ===")
        print(f"Final dataset: {len(df_filtered)} POIs")
        print(f"Kabupaten: {kabupaten_name}")
        print(f"Latitude range: {lat_range[0]:.4f} to {lat_range[1]:.4f}")
        print(f"Longitude range: {lon_range[0]:.4f} to {lon_range[1]:.4f}")
        if categories_of_interest:
            category_counts = df_filtered['category'].value_counts()
            print(f"Category distribution:\n{category_counts}")
        print("=" * 30)
    
    return df_filtered

@dataclass
class POICluster:
    """Represents a cluster of POIs with centroid and metadata."""
    cluster_id: int
    centroid_lat: float
    centroid_lon: float
    poi_ids: List[str]  # Changed to str to handle 'node/...' format
    primary_categories: List[str]
    poi_count: int
    avg_density_score: float
    priority_score: float

class ScalablePOIRouter:
    """
    Multi-tiered routing system for large POI datasets.
    
    Workflow:
    1. Filter/Score POIs by business importance
    2. Cluster POIs geographically 
    3. Create cluster-level route (TSP)
    4. Create intra-cluster routes (A*)
    """
    
    def __init__(self, poi_df: pd.DataFrame):
        """
        Initialize router with preprocessed POI data.
        
        Parameters:
        poi_df: Preprocessed POI DataFrame (already filtered and cleaned)
        """
        self.poi_df = poi_df.copy()
        self.clusters: List[POICluster] = []
        
        print(f"Initialized router with {len(self.poi_df)} POIs")
    
    def calculate_poi_priority_scores(self, 
                                    category_weights: Dict[str, float] = None,
                                    density_radius_km: float = 2.0) -> pd.DataFrame:
        """
        Step 1: Calculate priority scores for POIs based on category importance and local density.
        
        Parameters:
        category_weights: Dictionary mapping category -> importance weight
        density_radius_km: Radius for calculating local density
        
        Returns:
        DataFrame with added priority_score column
        """
        if category_weights is None:
            # Default weights - adjust based on your business priorities
            category_weights = {
                'mall': 10.0,
                'supermarket': 8.0,
                'office': 7.0,
                'bank': 6.0,
                'restaurant': 5.0,
                'convenience': 4.0,
                'school': 3.0,
                'hospital': 8.0,
                'government': 6.0,
                'shop': 4.0,
                'fuel': 5.0
            }
        
        df = self.poi_df.copy()
        
        # Base category score
        df['category_score'] = df['category'].map(category_weights).fillna(3.0)
        
        # Calculate local density score (number of POIs within radius)
        print("Calculating density scores...")
        density_scores = []
        
        coordinates = df[['latitude', 'longitude']].values
        n_pois = len(df)
        
        for idx in range(n_pois):
            current_lat, current_lon = coordinates[idx]
            
            # Vectorized distance calculation for efficiency
            lat_diff = np.abs(coordinates[:, 0] - current_lat) * 111  # ~111 km per degree
            lon_diff = np.abs(coordinates[:, 1] - current_lon) * 85   # ~85 km per degree at Indonesia latitude
            approximate_distance = np.sqrt(lat_diff**2 + lon_diff**2)
            
            nearby_count = np.sum(approximate_distance <= density_radius_km)
            density_scores.append(nearby_count)
            
            if (idx + 1) % 100 == 0 or idx == n_pois - 1:
                print(f"Processed {idx + 1}/{n_pois} POIs for density calculation")
        
        df['density_score'] = density_scores
        
        # Normalize density scores (0-10 scale)
        max_density = df['density_score'].max() if df['density_score'].max() > 0 else 1
        df['density_score_normalized'] = (df['density_score'] / max_density) * 10
        
        # Combined priority score
        df['priority_score'] = (df['category_score'] * 0.7) + (df['density_score_normalized'] * 0.3)
        
        self.poi_df = df
        return df
    
    def create_poi_clusters(self, 
                          max_clusters: int = 20,
                          min_samples_per_cluster: int = 5,
                          method: str = 'kmeans',
                          priority_percentile: float = 0.7) -> List[POICluster]:
        """
        Step 2: Create geographical clusters of POIs.
        
        Parameters:
        max_clusters: Maximum number of clusters to create
        min_samples_per_cluster: Minimum POIs per cluster
        method: 'kmeans' or 'dbscan'
        priority_percentile: Use top X% of POIs by priority for clustering
        
        Returns:
        List of POICluster objects
        """
        if 'priority_score' not in self.poi_df.columns:
            print("Priority scores not calculated. Running calculation first...")
            self.calculate_poi_priority_scores()
        
        # Filter to high-priority POIs only
        priority_threshold = self.poi_df['priority_score'].quantile(priority_percentile)
        high_priority_pois = self.poi_df[self.poi_df['priority_score'] >= priority_threshold].copy()
        
        print(f"Using {len(high_priority_pois)} high-priority POIs for clustering (top {int((1-priority_percentile)*100)}%)")
        
        if len(high_priority_pois) < min_samples_per_cluster:
            print(f"Warning: Only {len(high_priority_pois)} high-priority POIs found. Using all POIs for clustering.")
            high_priority_pois = self.poi_df.copy()
        
        # Prepare coordinates for clustering
        coordinates = high_priority_pois[['latitude', 'longitude']].values
        
        if method == 'kmeans':
            # K-means clustering
            n_clusters = min(max_clusters, len(high_priority_pois) // min_samples_per_cluster)
            n_clusters = max(2, n_clusters)  # At least 2 clusters
            
            print(f"Using K-means with {n_clusters} clusters")
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(coordinates)
            
        else:  # DBSCAN
            # DBSCAN clustering - automatically determines number of clusters
            # eps in degrees (approximately 2km in Indonesia)
            eps_degrees = 2.0 / 111.0  # Convert km to degrees
            
            print(f"Using DBSCAN with eps={eps_degrees:.4f} degrees, min_samples={min_samples_per_cluster}")
            dbscan = DBSCAN(eps=eps_degrees, min_samples=min_samples_per_cluster)
            cluster_labels = dbscan.fit_predict(coordinates)
        
        # Create POICluster objects
        clusters = []
        unique_labels = np.unique(cluster_labels)
        
        for cluster_id in unique_labels:
            if cluster_id == -1:  # DBSCAN noise points
                continue
                
            cluster_pois = high_priority_pois[cluster_labels == cluster_id]
            
            if len(cluster_pois) < min_samples_per_cluster:
                continue
            
            # Calculate cluster centroid
            centroid_lat = cluster_pois['latitude'].mean()
            centroid_lon = cluster_pois['longitude'].mean()
            
            # Get primary categories in this cluster
            category_counts = cluster_pois['category'].value_counts()
            primary_categories = category_counts.head(3).index.tolist()
            
            # Calculate average priority score
            avg_priority = cluster_pois['priority_score'].mean()
            
            cluster = POICluster(
                cluster_id=int(cluster_id),
                centroid_lat=centroid_lat,
                centroid_lon=centroid_lon,
                poi_ids=cluster_pois['id'].tolist(),
                primary_categories=primary_categories,
                poi_count=len(cluster_pois),
                avg_density_score=cluster_pois['density_score'].mean(),
                priority_score=avg_priority
            )
            
            clusters.append(cluster)
        
        # Sort clusters by priority score (descending)
        clusters.sort(key=lambda x: x.priority_score, reverse=True)
        
        self.clusters = clusters
        print(f"Created {len(clusters)} clusters")
        
        # Print cluster summary
        for i, cluster in enumerate(clusters[:5]):  # Show top 5 clusters
            print(f"  Cluster {cluster.cluster_id}: {cluster.poi_count} POIs, "
                  f"categories: {cluster.primary_categories}, "
                  f"priority: {cluster.priority_score:.2f}")
        
        return clusters
    
    def solve_cluster_level_tsp(self, 
                              selected_clusters: List[POICluster],
                              start_cluster_id: Optional[int] = None,
                              end_cluster_id: Optional[int] = None) -> Tuple[List[int], float]:
        """
        Step 3: Solve TSP for cluster-level routing.
        
        Parameters:
        selected_clusters: List of clusters to visit
        start_cluster_id: Optional starting cluster
        end_cluster_id: Optional ending cluster
        
        Returns:
        Tuple of (cluster_order, total_distance)
        """
        if len(selected_clusters) <= 2:
            return [c.cluster_id for c in selected_clusters], 0.0
        
        # Create distance matrix between cluster centroids
        n = len(selected_clusters)
        distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    c1, c2 = selected_clusters[i], selected_clusters[j]
                    distance_matrix[i][j] = haversine_distance(
                        c1.centroid_lat, c1.centroid_lon,
                        c2.centroid_lat, c2.centroid_lon
                    )
        
        if HAS_ORTOOLS and n > 3:
            # Use OR-Tools for optimal TSP solution
            try:
                order, distance = self._solve_tsp_ortools(distance_matrix, selected_clusters,
                                                        start_cluster_id, end_cluster_id)
                return order, distance
            except Exception as e:
                print(f"OR-Tools TSP failed: {e}. Using nearest neighbor approximation.")
        
        # Fallback: Nearest neighbor approximation
        return self._solve_tsp_nearest_neighbor(distance_matrix, selected_clusters,
                                              start_cluster_id, end_cluster_id)
    
    def _solve_tsp_ortools(self, distance_matrix: np.ndarray, 
                          clusters: List[POICluster],
                          start_id: Optional[int], 
                          end_id: Optional[int]) -> Tuple[List[int], float]:
        """Solve TSP using OR-Tools."""
        n = len(clusters)
        
        # Convert distance matrix to integer (OR-Tools requirement)
        int_distance_matrix = (distance_matrix * 1000).astype(int)  # Convert to meters
        
        # Create routing model
        manager = pywrapcp.RoutingIndexManager(n, 1, 0)  # 1 vehicle, start at index 0
        routing = pywrapcp.RoutingModel(manager)
        
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int_distance_matrix[from_node][to_node]
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Set search parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        search_parameters.time_limit.seconds = 10  # Limit solving time
        
        # Solve
        solution = routing.SolveWithParameters(search_parameters)
        
        if solution:
            route_order = []
            total_distance = 0
            index = routing.Start(0)
            
            while not routing.IsEnd(index):
                route_order.append(clusters[manager.IndexToNode(index)].cluster_id)
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                if not routing.IsEnd(index):
                    total_distance += distance_matrix[manager.IndexToNode(previous_index)][manager.IndexToNode(index)]
            
            return route_order, total_distance
        else:
            raise ValueError("No solution found")
    
    def _solve_tsp_nearest_neighbor(self, distance_matrix: np.ndarray,
                                  clusters: List[POICluster],
                                  start_id: Optional[int],
                                  end_id: Optional[int]) -> Tuple[List[int], float]:
        """Solve TSP using nearest neighbor heuristic (kept for OR-Tools compatibility)."""
        n = len(clusters)
        
        # Start from specified cluster or first cluster
        if start_id is not None:
            start_idx = next(i for i, c in enumerate(clusters) if c.cluster_id == start_id)
        else:
            start_idx = 0
        
        unvisited = set(range(n))
        current = start_idx
        route = [clusters[current].cluster_id]
        total_distance = 0.0
        unvisited.remove(current)
        
        while unvisited:
            # Find nearest unvisited cluster
            nearest_idx = min(unvisited, key=lambda x: distance_matrix[current][x])
            
            total_distance += distance_matrix[current][nearest_idx]
            current = nearest_idx
            route.append(clusters[current].cluster_id)
            unvisited.remove(current)
        
        return route, total_distance
    
    def create_intra_cluster_route(self, cluster: POICluster, 
                                 max_pois_per_cluster: int = 5,
                                 use_astar: bool = True) -> Tuple[List[str], float]:
        """
        Step 4: Create optimal route within a cluster using A* pathfinding.
        
        Parameters:
        cluster: POICluster to route within
        max_pois_per_cluster: Maximum POIs to visit within the cluster
        use_astar: Whether to use A* for pathfinding (default True)
        
        Returns:
        Tuple of (poi_order, total_distance)
        """
        cluster_pois = self.poi_df[self.poi_df['id'].isin(cluster.poi_ids)].copy()
        
        # Select top POIs by priority score
        top_pois = cluster_pois.nlargest(max_pois_per_cluster, 'priority_score')
        
        if len(top_pois) <= 1:
            return top_pois['id'].tolist(), 0.0
        
        if len(top_pois) == 2:
            poi_ids = top_pois['id'].tolist()
            lat1, lon1 = top_pois.iloc[0]['latitude'], top_pois.iloc[0]['longitude']
            lat2, lon2 = top_pois.iloc[1]['latitude'], top_pois.iloc[1]['longitude']
            distance = haversine_distance(lat1, lon1, lat2, lon2)
            return poi_ids, distance
        
        if use_astar:
            return self._create_intra_cluster_route_astar(top_pois)
        else:
            return self._create_intra_cluster_route_nearest_neighbor(top_pois)
    
    def _create_intra_cluster_route_astar(self, poi_df: pd.DataFrame) -> Tuple[List[str], float]:
        """Create intra-cluster route using A* pathfinding."""
        # Create nodes dictionary for A* pathfinder
        poi_nodes = {}
        for _, poi in poi_df.iterrows():
            poi_nodes[poi['id']] = (poi['latitude'], poi['longitude'])
        
        # Initialize A* pathfinder
        pathfinder = AStarPathfinder(poi_nodes)
        
        # Get POI IDs sorted by priority (highest first)
        poi_ids = poi_df.sort_values('priority_score', ascending=False)['id'].tolist()
        
        try:
            # Find optimal tour using A*
            route_path, total_distance = pathfinder.find_optimal_tour(poi_ids)
            return route_path, total_distance
        except Exception as e:
            print(f"A* intra-cluster routing failed: {e}. Using nearest neighbor fallback.")
            return self._create_intra_cluster_route_nearest_neighbor(poi_df)
    
    def _create_intra_cluster_route_nearest_neighbor(self, poi_df: pd.DataFrame) -> Tuple[List[str], float]:
        """Fallback nearest neighbor routing for intra-cluster."""
        poi_ids = poi_df['id'].tolist()
        coordinates = [(row['latitude'], row['longitude']) for _, row in poi_df.iterrows()]
        
        # Create distance matrix
        n = len(coordinates)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    distances[i][j] = haversine_distance(
                        coordinates[i][0], coordinates[i][1],
                        coordinates[j][0], coordinates[j][1]
                    )
        
        # Nearest neighbor starting from highest priority POI (index 0)
        current = 0
        unvisited = set(range(1, n))
        route = [poi_ids[current]]
        total_distance = 0.0
        
        while unvisited:
            nearest = min(unvisited, key=lambda x: distances[current][x])
            total_distance += distances[current][nearest]
            current = nearest
            route.append(poi_ids[current])
            unvisited.remove(current)
        
        return route, total_distance

def generate_optimal_route(csv_path: str, 
                         kabupaten_name: str,
                         categories_of_interest: Optional[List[str]] = None,
                         max_clusters_per_day: int = 8,
                         max_pois_per_cluster: int = 4,
                         category_weights: Dict[str, float] = None,
                         clustering_method: str = 'kmeans',
                         verbose: bool = True) -> Dict:
    """
    Main function to generate optimal marketing route from CSV file.
    
    Parameters:
    csv_path: Path to the CSV file (e.g., 'test_path.csv')
    kabupaten_name: Name of kabupaten to analyze (e.g., 'Kota Bandung')
    categories_of_interest: Optional list of categories to include (e.g., ['mall', 'supermarket'])
    max_clusters_per_day: Maximum cluster areas to visit per day
    max_pois_per_cluster: Maximum POIs to visit within each cluster
    category_weights: Optional custom category importance weights
    clustering_method: 'kmeans' or 'dbscan'
    verbose: Whether to print detailed progress information
    
    Returns:
    Dictionary containing complete route information
    """
    try:
        print(f"\n=== Generating Optimal Marketing Route ===")
        print(f"Target: {kabupaten_name}")
        if categories_of_interest:
            print(f"Categories: {categories_of_interest}")
        
        # Step 1: Load and preprocess data
        poi_df = load_and_preprocess_data(
            csv_path=csv_path,
            kabupaten_name=kabupaten_name,
            categories_of_interest=categories_of_interest,
            verbose=verbose
        )
        
        if len(poi_df) == 0:
            return {"error": "No POIs remaining after preprocessing"}
        
        # Check if we have enough POIs for clustering
        if len(poi_df) < 10:
            print(f"Warning: Only {len(poi_df)} POIs found. Route may be very simple.")
        
        # Step 2: Initialize router and calculate priority scores
        router = ScalablePOIRouter(poi_df)
        
        if verbose:
            print("\nCalculating POI priority scores...")
        router.calculate_poi_priority_scores(category_weights)
        
        # Step 3: Create clusters
        if verbose:
            print("\nCreating POI clusters...")
        clusters = router.create_poi_clusters(
            max_clusters=max_clusters_per_day * 2,  # Create more for selection
            method=clustering_method
        )
        
        if len(clusters) == 0:
            return {"error": "No valid clusters created"}
        
        # Step 4: Select top clusters for daily route
        selected_clusters = clusters[:max_clusters_per_day]
        if verbose:
            print(f"\nSelected {len(selected_clusters)} clusters for daily route")
        
        # Step 5: Solve cluster-level TSP
        if verbose:
            print("\nOptimizing cluster-level route...")
        cluster_order, cluster_route_distance = router.solve_cluster_level_tsp(selected_clusters)
        
        # Step 6: Create detailed route with intra-cluster paths
        if verbose:
            print("\nCreating detailed intra-cluster routes...")
        
        detailed_route = []
        total_detailed_distance = cluster_route_distance
        route_segments = []
        
        for cluster_id in cluster_order:
            cluster = next(c for c in selected_clusters if c.cluster_id == cluster_id)
            poi_route, intra_distance = router.create_intra_cluster_route(cluster, max_pois_per_cluster)
            
            # Get full POI data
            segment_pois = []
            for poi_id in poi_route:
                poi_data = router.poi_df[router.poi_df['id'] == poi_id].iloc[0].to_dict()
                segment_pois.append(poi_data)
                detailed_route.append(poi_data)
            
            route_segments.append({
                'cluster_id': cluster_id,
                'cluster_centroid': (cluster.centroid_lat, cluster.centroid_lon),
                'primary_categories': cluster.primary_categories,
                'pois_in_segment': segment_pois,
                'segment_distance_km': round(intra_distance, 2),
                'segment_poi_count': len(segment_pois)
            })
            
            total_detailed_distance += intra_distance
        
        # Calculate summary statistics
        total_pois = len(detailed_route)
        estimated_time = total_detailed_distance / 40.0  # Assume 40 km/h average speed
        
        if verbose:
            print(f"\n=== ROUTE GENERATION COMPLETE ===")
            print(f"Total POIs to visit: {total_pois}")
            print(f"Total distance: {total_detailed_distance:.2f} km")
            print(f"Estimated travel time: {estimated_time:.1f} hours")
            print(f"Clusters visited: {len(route_segments)}")
        
        return {
            'success': True,
            'kabupaten': kabupaten_name,
            'categories_used': categories_of_interest,
            'route_summary': {
                'total_pois': total_pois,
                'total_distance_km': round(total_detailed_distance, 2),
                'estimated_travel_time_hours': round(estimated_time, 1),
                'clusters_visited': len(route_segments),
                'cluster_route_distance_km': round(cluster_route_distance, 2)
            },
            'route_segments': route_segments,
            'detailed_route': detailed_route,
            'cluster_summary': [
                {
                    'cluster_id': c.cluster_id,
                    'priority_score': round(c.priority_score, 2),
                    'poi_count': c.poi_count,
                    'primary_categories': c.primary_categories,
                    'centroid': (c.centroid_lat, c.centroid_lon)
                }
                for c in selected_clusters
            ],
            'data_summary': {
                'original_poi_count': len(poi_df),
                'clusters_created': len(clusters),
                'preprocessing_successful': True
            }
        }
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        if verbose:
            traceback.print_exc()
        
        return {
            'success': False,
            'error': str(e),
            'kabupaten': kabupaten_name,
            'categories_used': categories_of_interest
        }

def demo_with_sample_csv():
    """
    Demonstration function that creates a sample CSV file and tests the routing.
    Use this if you don't have the actual test_path.csv yet.
    """
    print("=== Creating Sample CSV for Demo ===")
    
    # Create sample data matching your CSV structure
    np.random.seed(42)
    n_sample = 100
    
    # Generate realistic POI data for Kota Bandung
    base_lat, base_lon = -6.9175, 107.6191  # Bandung coordinates
    
    categories = ['mall', 'supermarket', 'convenience', 'restaurant', 'office', 'bank', 'school', 'hospital']
    
    sample_data = []
    for i in range(n_sample):
        # Create node IDs like in your data
        node_id = f"node/{29391283 + i}"
        
        # Scattered around Bandung (roughly 10km radius)
        lat_offset = np.random.normal(0, 0.05)  # ~5km standard deviation
        lon_offset = np.random.normal(0, 0.05)
        
        sample_data.append({
            'id': node_id,
            'kabupaten_kota': 'Kota Bandung',
            'name': f'POI_{i}',
            'category': np.random.choice(categories),
            'latitude': base_lat + lat_offset,
            'longitude': base_lon + lon_offset
        })
    
    # Add some missing coordinate examples
    sample_data.append({
        'id': 'node/missing_coords_1',
        'kabupaten_kota': 'Kota Bandung',
        'name': 'Missing_POI_1',
        'category': 'mall',
        'latitude': np.nan,
        'longitude': 107.6191
    })
    
    sample_data.append({
        'id': 'node/missing_coords_2',
        'kabupaten_kota': 'Kota Bandung', 
        'name': 'Missing_POI_2',
        'category': 'office',
        'latitude': -6.9175,
        'longitude': np.nan
    })
    
    # Add some POIs from different kabupaten
    for i in range(10):
        sample_data.append({
            'id': f'node/jakarta_{i}',
            'kabupaten_kota': 'Jakarta Pusat',
            'name': f'Jakarta_POI_{i}',
            'category': np.random.choice(categories),
            'latitude': -6.1751 + np.random.normal(0, 0.02),
            'longitude': 106.8650 + np.random.normal(0, 0.02)
        })
    
    # Create DataFrame and save to CSV
    sample_df = pd.DataFrame(sample_data)
    sample_csv_path = 'sample_test_path.csv'
    sample_df.to_csv(sample_csv_path, index=False)
    
    print(f"✓ Created sample CSV: {sample_csv_path}")
    print(f"  Total rows: {len(sample_df)}")
    print(f"  Bandung POIs: {len(sample_df[sample_df['kabupaten_kota'] == 'Kota Bandung'])}")
    print(f"  Categories: {list(sample_df['category'].unique())}")
    
    # Test the routing function
    print(f"\n=== Testing Route Generation ===")
    
    # Test 1: All categories
    print("\nTest 1: All categories")
    result1 = generate_optimal_route(
        csv_path=sample_csv_path,
        kabupaten_name='Kota Bandung',
        categories_of_interest=None,
        max_clusters_per_day=5,
        max_pois_per_cluster=3,
        verbose=True
    )
    
    if result1['success']:
        summary = result1['route_summary']
        print(f"✓ Route generated successfully!")
        print(f"  Total POIs: {summary['total_pois']}")
        print(f"  Total distance: {summary['total_distance_km']} km")
        print(f"  Clusters: {summary['clusters_visited']}")
    else:
        print(f"✗ Route generation failed: {result1['error']}")
    
    # Test 2: Specific categories only
    print(f"\nTest 2: Specific categories (mall, supermarket, office)")
    result2 = generate_optimal_route(
        csv_path=sample_csv_path,
        kabupaten_name='Kota Bandung',
        categories_of_interest=['mall', 'supermarket', 'office'],
        max_clusters_per_day=4,
        max_pois_per_cluster=2,
        verbose=False  # Less verbose for second test
    )
    
    if result2['success']:
        summary = result2['route_summary']
        print(f"✓ Filtered route generated!")
        print(f"  Total POIs: {summary['total_pois']}")
        print(f"  Categories used: {result2['categories_used']}")
        
        # Show route details
        print(f"\n  Route segments:")
        for i, segment in enumerate(result2['route_segments']):
            poi_names = [poi['name'] for poi in segment['pois_in_segment']]
            print(f"    {i+1}. Cluster {segment['cluster_id']}: {poi_names}")
    else:
        print(f"✗ Filtered route failed: {result2['error']}")
    
    # Test 3: Error handling - invalid kabupaten
    print(f"\nTest 3: Error handling (invalid kabupaten)")
    result3 = generate_optimal_route(
        csv_path=sample_csv_path,
        kabupaten_name='Nonexistent City',
        verbose=False
    )
    
    if not result3['success']:
        print(f"✓ Error correctly handled: {result3['error']}")
    else:
        print(f"✗ Should have failed but didn't")
    
    return sample_csv_path

# Usage examples and instructions
def print_usage_instructions():
    """Print comprehensive usage instructions."""
    print("""
=== USAGE INSTRUCTIONS ===

1. BASIC USAGE WITH YOUR CSV FILE:
   
   result = generate_optimal_route(
       csv_path='test_path.csv',
       kabupaten_name='Kota Bandung'
   )
   
2. FILTER BY CATEGORIES:
   
   result = generate_optimal_route(
       csv_path='test_path.csv',
       kabupaten_name='Kota Bandung',
       categories_of_interest=['mall', 'supermarket', 'office', 'bank']
   )
   
3. CUSTOMIZE ROUTE PARAMETERS:
   
   result = generate_optimal_route(
       csv_path='test_path.csv',
       kabupaten_name='Kota Bandung',
       categories_of_interest=['mall', 'supermarket'],
       max_clusters_per_day=6,        # Visit 6 geographic areas
       max_pois_per_cluster=3,        # 3 POIs per area = 18 total
       clustering_method='dbscan',    # or 'kmeans'
       verbose=True
   )

4. CUSTOM CATEGORY PRIORITIES:
   
   custom_weights = {
       'mall': 10.0,
       'office': 8.0,
       'supermarket': 7.0,
       'bank': 6.0,
       'restaurant': 4.0
   }
   
   result = generate_optimal_route(
       csv_path='test_path.csv',
       kabupaten_name='Kota Bandung',
       category_weights=custom_weights
   )

5. ACCESS RESULTS:
   
   if result['success']:
       # Route summary
       summary = result['route_summary']
       print(f"Total POIs: {summary['total_pois']}")
       print(f"Distance: {summary['total_distance_km']} km")
       print(f"Time: {summary['estimated_travel_time_hours']} hours")
       
       # Detailed route
       for poi in result['detailed_route']:
           print(f"{poi['name']} ({poi['category']})")
       
       # Route segments
       for segment in result['route_segments']:
           print(f"Cluster {segment['cluster_id']}: {segment['primary_categories']}")
   else:
       print(f"Error: {result['error']}")

=== REQUIRED CSV STRUCTURE ===
Your 'test_path.csv' must have these columns:
- id: Unique identifier (e.g., 'node/29391283')  
- latitude: Decimal degrees (required, no missing values after preprocessing)
- longitude: Decimal degrees (required, no missing values after preprocessing)
- category: POI type (e.g., 'mall', 'supermarket')
- kabupaten_kota: Region name (e.g., 'Kota Bandung')

=== PREPROCESSING STEPS ===
The function automatically:
1. ✓ Drops rows with missing latitude/longitude
2. ✓ Filters to specified kabupaten only  
3. ✓ Filters to specified categories (if provided)
4. ✓ Validates coordinate data types
5. ✓ Checks coordinate ranges for Indonesia

=== PARAMETERS GUIDE ===
- max_clusters_per_day: 4-10 recommended (geographic areas to visit)
- max_pois_per_cluster: 2-5 recommended (POIs per area)  
- categories_of_interest: Filter to high-value POI types
- clustering_method: 'kmeans' (fixed clusters) vs 'dbscan' (density-based)
- category_weights: Higher values = higher priority (scale 1-10)
""")

if __name__ == "__main__":
    print("=== Scalable POI Routing System ===")
    print("Ready to process your test_path.csv file!")
    print()
    
    # Check if the actual CSV file exists
    if os.path.exists('test_path.csv'):
        print("✓ Found test_path.csv in working directory")
        print("\nTo run analysis, use:")
        print("result = generate_optimal_route('test_path.csv', 'Kota Bandung')")
        print()
        print_usage_instructions()
    else:
        print("test_path.csv not found. Running demo with sample data...")
        sample_file = demo_with_sample_csv()
        print(f"\n✓ Demo completed! Sample file created: {sample_file}")
        print("\nReplace 'sample_test_path.csv' with 'test_path.csv' for real data.")
        print()
        print_usage_instructions()
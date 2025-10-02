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
warnings.filterwarnings('ignore')

# Try to import OR-Tools for TSP (optional)
try:
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp
    HAS_ORTOOLS = True
except ImportError:
    HAS_ORTOOLS = False
    print("OR-Tools not available. Will use simplified TSP approximation.")

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate Haversine distance between two points in kilometers."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    return 6371.0 * 2 * math.asin(math.sqrt(a))

@dataclass
class POICluster:
    """Represents a cluster of POIs with centroid and metadata."""
    cluster_id: int
    centroid_lat: float
    centroid_lon: float
    poi_ids: List[int]
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
    
    def __init__(self, poi_df: pd.DataFrame, kabupaten: str):
        """
        Initialize router for a specific kabupaten.
        
        Parameters:
        poi_df: Full POI DataFrame
        kabupaten: Specific kabupaten to process
        """
        self.kabupaten = kabupaten
        self.poi_df = poi_df[poi_df['kabupaten_kota'] == kabupaten].copy().reset_index(drop=True)
        self.clusters: List[POICluster] = []
        
        print(f"Initialized router for {kabupaten} with {len(self.poi_df)} POIs")
    
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
                'government': 6.0
            }
        
        df = self.poi_df.copy()
        
        # Base category score
        df['category_score'] = df['category'].map(category_weights).fillna(2.0)
        
        # Calculate local density score (number of POIs within radius)
        print("Calculating density scores...")
        density_scores = []
        
        for idx, poi in df.iterrows():
            # Count nearby POIs (simplified - using approximate distance)
            lat_diff = np.abs(df['latitude'] - poi['latitude']) * 111  # ~111 km per degree
            lon_diff = np.abs(df['longitude'] - poi['longitude']) * 85   # ~85 km per degree at Indonesia latitude
            approximate_distance = np.sqrt(lat_diff**2 + lon_diff**2)
            
            nearby_count = np.sum(approximate_distance <= density_radius_km)
            density_scores.append(nearby_count)
            
            if idx % 500 == 0:
                print(f"Processed {idx}/{len(df)} POIs for density calculation")
        
        df['density_score'] = density_scores
        
        # Normalize density scores (0-10 scale)
        max_density = df['density_score'].max()
        df['density_score_normalized'] = (df['density_score'] / max_density) * 10
        
        # Combined priority score
        df['priority_score'] = (df['category_score'] * 0.7) + (df['density_score_normalized'] * 0.3)
        
        self.poi_df = df
        return df
    
    def create_poi_clusters(self, 
                          max_clusters: int = 20,
                          min_samples_per_cluster: int = 5,
                          method: str = 'kmeans') -> List[POICluster]:
        """
        Step 2: Create geographical clusters of POIs.
        
        Parameters:
        max_clusters: Maximum number of clusters to create
        min_samples_per_cluster: Minimum POIs per cluster
        method: 'kmeans' or 'dbscan'
        
        Returns:
        List of POICluster objects
        """
        if 'priority_score' not in self.poi_df.columns:
            print("Priority scores not calculated. Running calculation first...")
            self.calculate_poi_priority_scores()
        
        # Filter to high-priority POIs only (top 30% by priority score)
        priority_threshold = self.poi_df['priority_score'].quantile(0.7)
        high_priority_pois = self.poi_df[self.poi_df['priority_score'] >= priority_threshold].copy()
        
        print(f"Using {len(high_priority_pois)} high-priority POIs for clustering")
        
        # Prepare coordinates for clustering
        coordinates = high_priority_pois[['latitude', 'longitude']].values
        
        if method == 'kmeans':
            # K-means clustering
            n_clusters = min(max_clusters, len(high_priority_pois) // min_samples_per_cluster)
            n_clusters = max(2, n_clusters)  # At least 2 clusters
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(coordinates)
            
        else:  # DBSCAN
            # DBSCAN clustering - automatically determines number of clusters
            # eps in degrees (approximately 2km in Indonesia)
            eps_degrees = 2.0 / 111.0  # Convert km to degrees
            
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
        cluster_id_to_idx = {c.cluster_id: i for i, c in enumerate(selected_clusters)}
        
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
                total_distance += distance_matrix[manager.IndexToNode(previous_index)][manager.IndexToNode(index)]
            
            return route_order, total_distance
        else:
            raise ValueError("No solution found")
    
    def _solve_tsp_nearest_neighbor(self, distance_matrix: np.ndarray,
                                  clusters: List[POICluster],
                                  start_id: Optional[int],
                                  end_id: Optional[int]) -> Tuple[List[int], float]:
        """Solve TSP using nearest neighbor heuristic."""
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
                                 max_pois_per_cluster: int = 5) -> Tuple[List[int], float]:
        """
        Step 4: Create optimal route within a cluster using A* or simple nearest neighbor.
        
        Parameters:
        cluster: POICluster to route within
        max_pois_per_cluster: Maximum POIs to visit within the cluster
        
        Returns:
        Tuple of (poi_order, total_distance)
        """
        cluster_pois = self.poi_df[self.poi_df['id'].isin(cluster.poi_ids)].copy()
        
        # Select top POIs by priority score
        top_pois = cluster_pois.nlargest(max_pois_per_cluster, 'priority_score')
        
        if len(top_pois) <= 2:
            return top_pois['id'].tolist(), 0.0
        
        # Use nearest neighbor for intra-cluster routing (simpler for small sets)
        poi_ids = top_pois['id'].tolist()
        coordinates = [(row['latitude'], row['longitude']) for _, row in top_pois.iterrows()]
        
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
        
        # Nearest neighbor starting from highest priority POI
        current = 0  # Start with first (highest priority) POI
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

def create_daily_marketing_route(poi_df: pd.DataFrame, 
                               kabupaten: str,
                               max_clusters_per_day: int = 8,
                               max_pois_per_cluster: int = 4,
                               category_preferences: Dict[str, float] = None) -> Dict:
    """
    Main function to create a practical daily marketing route for a kabupaten.
    
    Parameters:
    poi_df: Full POI DataFrame
    kabupaten: Target kabupaten
    max_clusters_per_day: Maximum cluster areas to visit per day
    max_pois_per_cluster: Maximum POIs to visit within each cluster
    category_preferences: Optional category weights for prioritization
    
    Returns:
    Dictionary containing complete route information
    """
    print(f"\n=== Creating Daily Marketing Route for {kabupaten} ===")
    
    # Initialize router
    router = ScalablePOIRouter(poi_df, kabupaten)
    
    if len(router.poi_df) == 0:
        return {"error": f"No POIs found for kabupaten: {kabupaten}"}
    
    # Step 1: Calculate priority scores
    print("\n1. Calculating POI priority scores...")
    router.calculate_poi_priority_scores(category_preferences)
    
    # Step 2: Create clusters
    print("\n2. Creating POI clusters...")
    clusters = router.create_poi_clusters(max_clusters=max_clusters_per_day * 2)  # Create more clusters for selection
    
    if len(clusters) == 0:
        return {"error": "No valid clusters created"}
    
    # Step 3: Select top clusters for daily route
    selected_clusters = clusters[:max_clusters_per_day]
    print(f"\n3. Selected {len(selected_clusters)} clusters for daily route")
    
    # Step 4: Solve cluster-level TSP
    print("\n4. Optimizing cluster-level route...")
    cluster_order, cluster_route_distance = router.solve_cluster_level_tsp(selected_clusters)
    
    # Step 5: Create detailed route with intra-cluster paths
    print("\n5. Creating detailed intra-cluster routes...")
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
            'segment_distance_km': round(intra_distance, 2)
        })
        
        total_detailed_distance += intra_distance
    
    # Calculate summary statistics
    total_pois = len(detailed_route)
    estimated_time = total_detailed_distance / 40.0  # Assume 40 km/h average speed
    
    print(f"\n=== Route Summary ===")
    print(f"Total POIs to visit: {total_pois}")
    print(f"Total distance: {total_detailed_distance:.2f} km")
    print(f"Estimated travel time: {estimated_time:.1f} hours")
    print(f"Clusters visited: {len(route_segments)}")
    
    return {
        'kabupaten': kabupaten,
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
        ]
    }

def demo_scalable_routing():
    """Demonstration with synthetic large dataset."""
    print("=== Scalable POI Routing Demo ===")
    
    # Create synthetic dataset mimicking your structure
    np.random.seed(42)
    kabupaten = "Kota Bandung"
    n_pois = 100  # Smaller demo dataset
    
    # Generate random POI data around Bandung
    base_lat, base_lon = -6.9, 107.6
    
    categories = ['mall', 'supermarket', 'convenience', 'restaurant', 'office', 'bank', 'school']
    category_weights = [0.1, 0.15, 0.3, 0.25, 0.1, 0.05, 0.05]
    
    # demo_data = {
    #     'id': range(1, n_pois + 1),
    #     'kabupaten_kota': [kabupaten] * n_pois,
    #     'name': [f'POI_{i}' for i in range(1, n_pois + 1)],
    #     'category': np.random.choice(categories, n_pois, p=category_weights),
    #     'latitude': np.random.normal(base_lat, 0.05, n_pois),  # ~5km spread
    #     'longitude': np.random.normal(base_lon, 0.05, n_pois)
    # }
    
    poi_df = pd.read_csv('../test_path.csv')
    
    print(f"Demo dataset: {len(poi_df)} POIs in {kabupaten}")
    print(f"Category distribution:\n{poi_df['category'].value_counts()}")
    
    # Run scalable routing
    try:
        route_info = create_daily_marketing_route(
            poi_df=poi_df,
            kabupaten=kabupaten,
            max_clusters_per_day=6,
            max_pois_per_cluster=3
        )
        
        if 'error' in route_info:
            print(f"Error: {route_info['error']}")
            return
        
        print("\n=== FINAL ROUTE RESULTS ===")
        summary = route_info['route_summary']
        print(f"✓ Total POIs in route: {summary['total_pois']}")
        print(f"✓ Total distance: {summary['total_distance_km']} km")  
        print(f"✓ Estimated time: {summary['estimated_travel_time_hours']} hours")
        print(f"✓ Cluster areas: {summary['clusters_visited']}")
        
        print(f"\n=== ROUTE SEGMENTS ===")
        for i, segment in enumerate(route_info['route_segments']):
            print(f"Segment {i+1}: Cluster {segment['cluster_id']}")
            print(f"  Categories: {', '.join(segment['primary_categories'])}")
            print(f"  POIs: {len(segment['pois_in_segment'])}")
            print(f"  POI Names: {[poi['name'] for poi in segment['pois_in_segment']]}")
            print(f"  Segment distance: {segment['segment_distance_km']} km")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demo_scalable_routing()
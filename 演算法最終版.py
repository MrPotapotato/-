import pandas as pd
from collections import deque, defaultdict
import matplotlib.pyplot as plt

# ---------------------- Core Utilities ----------------------
def bfs(path_dict, start, end):
    """Find shortest path using BFS."""
    dist = {start: 0}
    pred = defaultdict(list)
    queue = deque([start])
    
    while queue:
        current = queue.popleft()
        for neighbor in path_dict.get(current, []):
            if neighbor not in dist:
                dist[neighbor] = dist[current] + 1
                pred[neighbor].append(current)
                queue.append(neighbor)
            elif dist[neighbor] == dist[current] + 1:
                pred[neighbor].append(current)
    
    if end not in dist:
        return []
    
    path = [end]
    while path[-1] != start:
        path.append(pred[path[-1]][0])
    return path[::-1]

def calculate_manhattan(path, site_dict):
    """Calculate Manhattan distance from path directions."""
    x, y = 0, 0
    dir_map = {1: (0,1), 2: (1,0), 3: (0,-1), 4: (-1,0)}
    for i in range(len(path)-1):
        dir_num, _ = site_dict[path[i]][path[i+1]]
        dx, dy = dir_map[dir_num]
        x += dx
        y += dy
    return abs(x) + abs(y)

def generate_directions(path, site_dict):
    """Convert path to direction string."""
    if len(path) < 2:
        return ""
    
    directions = []
    initial_dir = site_dict[path[0]][path[1]][0]
    directions.append("f")
    
    for i in range(1, len(path)-1):
        current_node = path[i]
        next_node = path[i+1]
        final_dir = site_dict[current_node][next_node][0]
        
        turn_num = final_dir - initial_dir
        if turn_num == 0:
            turn_str = "f"
        elif abs(turn_num) == 2:
            turn_str = "b"
        elif turn_num == 1 or turn_num == -3:
            turn_str = "r"
        elif turn_num == 3 or turn_num == -1:
            turn_str = "l"
        directions.append(turn_str)
        initial_dir = final_dir
    
    return "".join(directions)

# ---------------------- Data Loading ----------------------
filepath = r"C:\Users\brian\Downloads\big_maze_113.csv"
raw_data = pd.read_csv(filepath)
site_dict = defaultdict(dict)
path_dict = defaultdict(list)

for _, row in raw_data.iterrows():
    site = int(row['index'])
    for dir_idx, direction in enumerate(['North', 'East', 'South', 'West']):
        neighbor = int(row[direction]) if pd.notna(row[direction]) else 0
        if neighbor > 0:
            dir_num = dir_idx + 1
            distance = int(row[[f"{d}D" for d in ['N','E','S','W']][dir_idx]])
            site_dict[site][neighbor] = (dir_num, distance)
            path_dict[site].append(neighbor)

# ---------------------- Common Setup ----------------------
end_points = [site for site, neighbors in path_dict.items() if len(neighbors) == 1]
start_site = int(input(f"Enter starting point (choose from {end_points}): "))

if start_site not in end_points:
    raise ValueError("Invalid starting point. Must be an endpoint.")

# Calculate scores for all endpoints
scores = {}
for ep in end_points:
    if ep == start_site:
        continue
    path = bfs(path_dict, start_site, ep)
    if path:
        scores[ep] = calculate_manhattan(path, site_dict) * 10

all_endpoints = set(scores.keys())

# ---------------------- Method Implementations ----------------------
def original_region_method():
    """Region-based collection through intersection grouping."""
    intersections = [site for site in path_dict if len(path_dict[site]) > 2]
    region_groups = defaultdict(list)
    
    # Group endpoints by nearest intersection
    for ep in all_endpoints:
        min_dist = float('inf')
        nearest_inter = None
        for inter in intersections:
            path = bfs(path_dict, ep, inter)
            if path and (len(path)-1 < min_dist):
                min_dist = len(path)-1
                nearest_inter = inter
        if nearest_inter:
            region_groups[nearest_inter].append(ep)
    
    # Calculate region efficiencies
    regions_data = []
    for inter, eps in region_groups.items():
        path = bfs(path_dict, start_site, inter)
        if not path:
            continue
        
        steps = len(path) - 1
        score = sum(scores[ep] for ep in eps)
        for ep in eps:
            ep_path = bfs(path_dict, inter, ep)
            if ep_path:
                steps += 2 * (len(ep_path)-1)
        
        if steps > 0:
            regions_data.append((inter, eps, score/steps))
    
    regions_data.sort(key=lambda x: -x[2])
    
    # Build path
    full_path = [start_site]
    current = start_site
    remaining = set(all_endpoints)
    
    for inter, eps, _ in regions_data:
        if not remaining:
            break
        
        # Move to intersection
        path = bfs(path_dict, current, inter)
        if not path:
            continue
        full_path.extend(path[1:])
        current = inter
        
        # Collect endpoints
        for ep in eps:
            if ep not in remaining:
                continue
            
            path = bfs(path_dict, current, ep)
            if path:
                full_path.extend(path[1:])
                current = ep
                remaining.remove(ep)
                
                # Return if more endpoints left
                if remaining:
                    path = bfs(path_dict, current, inter)
                    if path:
                        full_path.extend(path[1:])
                        current = inter
    
    return (full_path, generate_directions(full_path, site_dict), sum(scores[ep] for ep in (all_endpoints - remaining)))
        

def tsp_greedy_method():
    """Nearest-neighbor TSP approach."""
    full_path = [start_site]
    current = start_site
    remaining = set(all_endpoints)
    total_score = 0
    
    while remaining:
        # Find nearest endpoint
        nearest = None
        min_steps = float('inf')
        for ep in remaining:
            path = bfs(path_dict, current, ep)
            if path and (len(path)-1 < min_steps):
                min_steps = len(path)-1
                nearest = ep
        
        if not nearest:
            break
        
        # Move to nearest
        path = bfs(path_dict, current, nearest)
        full_path.extend(path[1:])
        current = nearest
        remaining.remove(nearest)
        total_score += scores[nearest]
    
    return full_path, generate_directions(full_path, site_dict), total_score


def score_ordered_method():
    """Highest-score-first approach."""
    full_path = [start_site]
    current = start_site
    remaining = sorted(scores.items(), key=lambda x: -x[1])
    collected = set()
    total_score = 0
    
    for ep, score in remaining:
        if ep in collected:
            continue
        
        path = bfs(path_dict, current, ep)
        if not path:
            continue
        
        full_path.extend(path[1:])
        current = ep
        collected.add(ep)
        total_score += score
    
    return full_path, generate_directions(full_path, site_dict), total_score


def efficiency_ordered_method():
    """Score-per-move efficiency approach."""
    full_path = [start_site]
    current = start_site
    remaining = set(all_endpoints)
    total_score = 0
    
    while remaining:
        best_ep = None
        max_eff = -1
        
        # Calculate efficiencies
        for ep in remaining:
            path = bfs(path_dict, current, ep)
            if not path:
                continue
            
            steps = len(path) - 1
            if steps == 0:
                continue
                
            efficiency = scores[ep] / steps
            if efficiency > max_eff:
                max_eff = efficiency
                best_ep = ep
        
        if not best_ep:
            break
        
        # Move to best
        path = bfs(path_dict, current, best_ep)
        full_path.extend(path[1:])
        current = best_ep
        remaining.remove(best_ep)
        total_score += scores[best_ep]
    
    return full_path, generate_directions(full_path, site_dict), total_score


# ---------------------- Analysis & Plotting ----------------------
def plot_cumulative(path, label):
    """Generate cumulative score plot for a path."""
    cum_scores = []
    current = 0
    visited = set()
    
    for node in path:
        if node in scores and node not in visited:
            current += scores[node]
            visited.add(node)
        cum_scores.append(current)
    
    plt.plot(cum_scores, label=label, alpha=0.8, linewidth=2)


# ---------------------- New Path Export Functions ----------------------
def create_path_dataframe(path, directions, site_dict, scores):
    """Create detailed move-by-move DataFrame for a path."""
    moves = []
    cumulative_score = 0
    visited = set()
    
    for i in range(len(path)-1):
        current = path[i]
        next_node = path[i+1]
        
        # Get direction and turn command
        direction = site_dict[current][next_node][0]
        turn_cmd = directions[i] if i < len(directions) else 'f'
        
        # Update score if visiting new endpoint
        if next_node in scores and next_node not in visited:
            cumulative_score += scores[next_node]
            visited.add(next_node)
        
        moves.append({
            'Move Number': i+1,
            'Current Site': current,
            'Direction': direction,
            'Turn Command': turn_cmd,
            'Next Site': next_node,
            'Cumulative Score': cumulative_score
        })
    
    return pd.DataFrame(moves)


# ---------------------- Main Execution with Export ----------------------
if __name__ == "__main__":
    import os
    from datetime import datetime
    # Run all methods
    methods = {
        "Region-Based": original_region_method,
        "TSP-Greedy": tsp_greedy_method,
        "Score-Order": score_ordered_method,
        "Efficiency-Order": efficiency_ordered_method
    }
    
    results = {}
    for name, method in methods.items():
        print(f"\n⚙️ Processing {name}...")
        try:
            results[name] = method()  # Stores (path, directions, score)
        except Exception as e:
            print(f"❌ Error in {name}: {str(e)}")
            continue

    # Print results in specified format
    print("\n=== Final Results ===")
    for name, (path, directions, score) in results.items():
        print(f"\n{name}:")
        print(f"Path: {path}")
        print(f"Directions: {directions}")
    
    # Save to text file with explicit path
    output_filename = "maze_solutions.txt"
    full_path = os.path.abspath(output_filename)
    
    try:
        with open(full_path, "w") as f:
            f.write("=== Maze Solutions ===\n")
            for name, (path, directions, _) in results.items():
                f.write(f"\n{name}:\n")
                f.write(f"Path: {path}\n")
                f.write(f"Directions: {directions}\n\n")
        
        print(f"\n✅ All solutions saved to: {full_path}")
        
        # Try to open the file automatically
        if os.name == 'nt':  # For Windows
            os.startfile(full_path)
        else:  # For Mac/Linux
            os.system(f'open "{full_path}"' if sys.platform == "darwin" else f'xdg-open "{full_path}"')
            
    except Exception as e:
        print(f"❌ Error saving file: {str(e)}")
    # Plot cumulative scores
    plt.figure(figsize=(12, 6))
    for name, (path, _, _) in results.items():
        cum_scores = []
        current = 0
        visited = set()
        for node in path:
            if node in scores and node not in visited:
                current += scores[node]
                visited.add(node)
            cum_scores.append(current)
        plt.plot(cum_scores, label=name)
    
    plt.title("Cumulative Score Comparison")
    plt.xlabel("Move Number")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.show()
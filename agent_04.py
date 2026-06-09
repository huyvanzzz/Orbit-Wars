"""
Orbit Wars -Enders FleetScocred >1000 on leaderboard 5/2026"

Cách đọc nhanh file này:
1. Các hằng số SUN_X, SUN_Y, SUN_RADIUS mô tả mặt trời ở giữa bản đồ.
2. Nhóm hàm hình học ở đầu file dùng để tính khoảng cách, thời gian bay,
   né mặt trời và dự đoán vị trí hành tinh đang quay.
3. Hàm agent(obs) là phần quan trọng nhất. Mỗi turn, game truyền obs vào;
   agent phân loại hành tinh/fleet, chọn chiến thuật hiện tại, rồi trả về
   danh sách move dạng [id_hành_tinh_gửi, góc_bắn, số_ship].

Ý tưởng tổng thể của bot:
- Nếu còn neutral gần thì ưu tiên mở rộng.
- Nếu đang mạnh hơn thì chuyển sang đánh enemy.
- Nếu bị đe dọa thì giữ quân hoặc phản công.
- Với hành tinh quay quanh mặt trời, bot không bắn vào vị trí hiện tại,
  mà ước lượng vị trí tương lai để bắn đón.
- Đường bay nào đi quá gần mặt trời sẽ bị bỏ qua hoặc chỉnh góc.
"""
import os
os.environ['KAGGLE_ENVELOPES'] = '0'

import math
ABC_SWITCH_STEP = 55

def _make_abc_agent():
    import math
    import kaggle_environments.envs.orbit_wars.orbit_wars as ow
    import numpy as np

    fleet_trajectories = []
    reinforcement_trajectories = []
    moving_planets = []
    planets_coords = {}
    steps = 0

    MAX_SPEED = 6.0
    # could use RL in future to tune these vars to optimal values
    MIN_SHIPS_MINE_ATTACK = 5
    MIN_SHIPS_TARGET_COOP_ATTACK = 20
    COOP_PLANET_CAP = 8
    COLLIDE_TICK_THOLD = 1

    FORMULA_DIST = 100
    FORMULA_PROD_MULT = 15
    FORMULA_ENEMY_BONUS_MULT = 10
    FORMULA_TOTAL_SHIPS_PERCENT = 0.7


    def get_custom_score(m, t):
        dist = math.sqrt((m.x - t.x)**2 + (m.y - t.y)**2)

        min_ships = t.ships + 1
        fleet_speed = 1.0 + (MAX_SPEED - 1.0) * (math.log(max(1, min_ships)) / math.log(1000)) ** 1.5
        eta = dist / fleet_speed

        enemy_produced = 0
        enemy_bonus = 0
        if t.owner != -1:
            enemy_produced = eta * t.production
            enemy_bonus = t.production

        total_ships = min_ships + enemy_produced

        # + close targets
        # + high production
        # + if planet is owned by enemy (capturing planet is more valuable because we gain ships, they lose ships)
        # - lot of enemies and enemies produced by arrival
        # - slow arrivals
        
        return (
            (FORMULA_DIST - dist)
            + (FORMULA_PROD_MULT * t.production)
            + (FORMULA_ENEMY_BONUS_MULT * enemy_bonus)
            - (FORMULA_TOTAL_SHIPS_PERCENT * total_ships)
            - (2 * eta)
        )

                

    def get_planets_under_attack(mine, fleets, player, vel):
        mov_pl_traj = {}
        under_attack = {}
        seen = set()
        fleets = [f for f in fleets if f.owner != player]
        for m in mine:
            if m.id in moving_planets:
                mov_pl_traj[m.id] = get_planet_trajectories(m, vel)

        for f in fleets:
            fleet_speed = 1.0 + (MAX_SPEED - 1.0) * (math.log(f.ships) / math.log(1000)) ** 1.5
            prev_x = f.x
            prev_y = f.y

            for tick in range(1, 61):
                next_x = f.x + math.cos(f.angle) * fleet_speed * tick
                next_y = f.y + math.sin(f.angle) * fleet_speed * tick

                for m in mine:
                    if m.id in moving_planets:
                        m_x, m_y = mov_pl_traj[m.id][tick-1] # tick is 1 based, index 0 based, so -1
                    else:
                        m_x, m_y = m.x, m.y
                        
                    if collides(prev_x, prev_y, next_x, next_y, m_x, m_y, m.radius): 
                        if (m.id, f.id) not in seen:
                            if m.id not in under_attack:
                                under_attack[m.id] = {
                                    "planet": m,
                                    "fleets": []
                                }
                                
                            under_attack[m.id]["fleets"].append({
                                "fleet": f,
                                "arrive_tick": tick
                            })
                            seen.add((m.id, f.id))
                
                prev_x = next_x
                prev_y = next_y            
                        
        return under_attack
        


    def refresh_local_obs(obs):
        planets = [ow.Planet(*p) for p in obs.get("planets", [])]
        mine = [p for p in planets if p.owner == obs.get("player", [])]
        targets = [p for p in planets if p.owner != obs.get("player", [])]
        player = obs.get("player", -2)
        fleets = [ow.Fleet(*f) for f in obs.get("fleets", [])]

        return {
            "planets": planets,
            "mine": mine,
            "targets": targets,
            "player": player,
            "fleets": fleets
        }

    def sun_collision(m, fleet_speed, angle, ticks=61):
        prev_x = m.x
        prev_y = m.y

        for tick in range(1, ticks):
            x = m.x + math.cos(angle) * fleet_speed * tick
            y = m.y + math.sin(angle) * fleet_speed * tick

            if collides(prev_x, prev_y, x, y, 50, 50, 10):
                return True

            prev_x = x
            prev_y = y
                
        return False


    def calculate_req_ships_moving(attacking_planets, t, base_ships, vel):
        MAX_SPEED = 6.0
        required_ships = base_ships
        planet_trajectories = get_planet_trajectories(t, vel)
        
        for _ in range(3):
            remainder = required_ships
            max_tick = 0

            for a_p in attacking_planets:
                p = a_p["planet"]
                p_ships = min(a_p["ships"], remainder)

                if p_ships > 0:
                    p_ships = min(a_p["ships"], max(p_ships, MIN_SHIPS_MINE_ATTACK))

                if p_ships <= 0:
                    continue
                
                fleet_speed = 1.0 + (MAX_SPEED - 1.0) * (math.log(max(1, p_ships)) / math.log(1000)) ** 1.5

                found_tick = 0
                for tick, (tx, ty) in enumerate(planet_trajectories, start=1):
                    dist = math.sqrt((p.x - tx)**2 + (p.y - ty)**2)
                    turns_to_arrive = math.floor(dist / fleet_speed)

                    if abs(turns_to_arrive - tick) <= 1:
                        found_tick = tick
                        break

                if found_tick > max_tick:
                    max_tick = found_tick

                remainder -= p_ships

            new_req = base_ships + (max_tick * t.production)
            if new_req == required_ships:
                break
            required_ships = new_req
            
        return required_ships

    def calculate_req_ships(attacking_planets, t, base_ships):
        required_ships = base_ships
        
        for _ in range(3):
            remainder = required_ships
            max_tick = 0
            
            for a_p in attacking_planets:
                p = a_p["planet"]
                p_ships = min(a_p["ships"], remainder)
                
                if p_ships > 0:
                    p_ships = min(a_p["ships"], max(p_ships, MIN_SHIPS_MINE_ATTACK))

                if p_ships <= 0:
                    continue
                
                ships_for_speed = max(1, p_ships)
                fleet_speed = 1.0 + (MAX_SPEED - 1.0) * (math.log(ships_for_speed) / math.log(1000)) ** 1.5
                dist = math.sqrt((p.x - t.x)**2 + (p.y - t.y)**2)
                tick_arrival = math.floor(dist / fleet_speed)
                
                if tick_arrival > max_tick:
                    max_tick = tick_arrival

                remainder -= p_ships

            new_req = base_ships + (max_tick * t.production)
            
            if new_req == required_ships:
                break
                
            required_ships = new_req
        
        return required_ships


    def calculate_angle(m, t):
        return math.atan2(t.y - m.y, t.x - m.x)
        

    def find_angle_to_moving_planet(p, t, ships, vel):
        fleet_speed = 1.0 + (MAX_SPEED - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5
        planet_trajectories = get_planet_trajectories(t, vel)

        for tick, (tx, ty) in enumerate(planet_trajectories, start=1):
            dx = tx - p.x
            dy = ty - p.y
            dist_to_target = math.sqrt(dx**2 + dy**2) - p.radius

            travel_dist = fleet_speed * tick
            miss_dist = abs(travel_dist - dist_to_target)

            if miss_dist > t.radius:
                continue
            
            angle = math.atan2(dy, dx)
            
            if sun_collision(p, fleet_speed, angle):
                return None, None

            return angle, tick

        return None, None


    def collides(x1, y1, x2, y2, cx, cy, r):
        vec_x = x2 - x1
        vec_y = y2 - y1

        vec_to_cx = cx - x1
        vec_to_cy = cy - y1

        vec_length_sq = vec_x**2 + vec_y**2

        if vec_length_sq == 0:
            dx = x1 - cx
            dy = y1 - cy
            return dx**2 + dy**2 <= r**2

        closest_point = (vec_to_cx * vec_x + vec_to_cy * vec_y) / vec_length_sq
        closest_point = max(0, min(1, closest_point))

        closest_x = x1 + closest_point * vec_x
        closest_y = y1 + closest_point * vec_y

        dx = closest_x - cx
        dy = closest_y - cy
        return dx**2 + dy**2 <= r**2


    def get_closest_planets_to_target(mine, t):
        planets = []
        for m in mine:
            dist = math.sqrt((m.x - t.x)**2 + (m.y - t.y)**2)
            planets.append((m, dist))
        planets = sorted(planets, key=lambda k: k[1])
        return planets
        

    def update_fleet_trajectories(fleets):
        for f_t in fleet_trajectories[:]:
            found = False
            for f in fleets:
                if f.from_planet_id == f_t["mine"].id and abs(f.angle - f_t["angle"]) < 1e-3:
                    found = True
                    break

            if found:
                f_t["arrive_tick"] = max(0, f_t["arrive_tick"] - 1)

            if not found:
                fleet_trajectories.remove(f_t)


    def update_reinforcement_trajectories(planets):
        planet_ids = {p.id for p in planets}
        
        for r_t in reinforcement_trajectories[:]:
            r_t["arrive_tick"] -= 1

            if r_t["arrive_tick"] <= 0:
                reinforcement_trajectories.remove(r_t)
                continue


    def get_planet_trajectories(p, vel):
        planet_trajectories = []
        angle = math.atan2(p.y - 50, p.x - 50)
        r = math.sqrt((p.x - 50)**2 + (p.y - 50)**2)
        for tick in range(1, 61): # max 60 ticks
            angle_t = angle + vel * tick
            x_t = 50 + r * math.cos(angle_t)
            y_t = 50 + r * math.sin(angle_t)
            planet_trajectories.append((x_t, y_t))

        return planet_trajectories
        

    def fill_moving_planets(obs):
        planets = [ow.Planet(*p) for p in obs.get("planets", [])]
        initial_by_id = {i[0]: ow.Planet(*i) for i in obs.get("initial_planets", [])}
        for p in planets:
            i = initial_by_id[p.id]
            if (p.x, p.y) != (i.x, i.y):
                if p.id not in moving_planets:
                    moving_planets.append(p.id)

    def get_reinforcement_plans(mine, under_attack):
        reinforcement_plans = {}
        
        for p in mine:
            if p.id in under_attack:
                attacking_fleets = sorted(
                    under_attack[p.id]["fleets"],
                    key=lambda att: att["arrive_tick"]
                )
                
                incoming_reinforcements = sorted(
                    [r for r in reinforcement_trajectories if r["target"].id == p.id],
                    key=lambda r: r["arrive_tick"]
                )
                
                p_available_ships = p.ships
                previous_tick = 0
                r_idx = 0

                for att in attacking_fleets:
                    att_arrive_tick = att["arrive_tick"]

                    p_available_ships += (att_arrive_tick - previous_tick) * p.production
                    
                    while (
                        r_idx < len(incoming_reinforcements)
                        and incoming_reinforcements[r_idx]["arrive_tick"] <= att_arrive_tick
                    ):
                        p_available_ships += incoming_reinforcements[r_idx]["total_ships"]
                        r_idx += 1

                    enemy_ships = att["fleet"].ships
                    p_available_ships -= enemy_ships
                    previous_tick = att_arrive_tick
                    
                    if p_available_ships < 0:
                        reinforcements_needed = max(MIN_SHIPS_MINE_ATTACK, abs(p_available_ships))
                        reinforcement_plans[p] = {
                            "ships_needed": reinforcements_needed,
                            "needed_by_tick": att_arrive_tick
                        }
                        break
                    
        return reinforcement_plans


    def agent(obs):
        nonlocal steps, fleet_trajectories, reinforcement_trajectories
        moves = []
        
        if steps < 2:
            steps += 1
            return []
        if steps == 2:
            fill_moving_planets(obs)
            steps = 3  

        lobs = refresh_local_obs(obs)
        update_fleet_trajectories(lobs.get("fleets", []))
        update_reinforcement_trajectories(lobs.get("planets", []))
        comet_planet_ids = obs.get("comet_planet_ids", [])    
        under_attack = get_planets_under_attack(lobs.get("mine", []), lobs.get("fleets", []), lobs.get("player", -2), obs.angular_velocity)
        exhausted_planets_id = set()
        
        if not lobs.get("targets", []):
            return []

        reinforcement_plans = get_reinforcement_plans(lobs.get("mine", []), under_attack)
        for p, plan in reinforcement_plans.items():
            already_reinforced = any(
                r["target"].id == p.id and r["arrive_tick"] >= 0
                for r in reinforcement_trajectories
            )

            if already_reinforced:
                continue
                
            ships_needed = plan["ships_needed"]
            needed_by_tick = plan["needed_by_tick"]
            nearest_planets = get_closest_planets_to_target(lobs.get("mine", []), p)
            
            for row in nearest_planets:
                p_np, _ = row
                
                if p_np.id == p.id or p_np.id in exhausted_planets_id:
                    continue

                p_np_available_ships = p_np.ships

                reserved_reinforcement_ships = sum(
                    r["total_ships"]
                    for r in reinforcement_trajectories
                    if r["mine"].id == p_np.id
                )
                
                p_np_available_ships -= reserved_reinforcement_ships

                if p_np.id in under_attack:
                    enemy_ships = sum(
                        att["fleet"].ships
                        for att in under_attack[p_np.id]["fleets"]
                    )
                    p_np_available_ships = max(0, p_np_available_ships - enemy_ships)

                sent_reinforcements = max(MIN_SHIPS_MINE_ATTACK, ships_needed)

                if p_np_available_ships < sent_reinforcements:
                    continue
                angle_np = None
                if p.id not in moving_planets:
                    angle_np = math.atan2(p.y - p_np.y, p.x - p_np.x)
                    fleet_speed = 1.0 + (MAX_SPEED - 1.0) * (math.log(sent_reinforcements) / math.log(1000)) ** 1.5                
                    dist = math.sqrt((p.x - p_np.x)**2 + (p.y - p_np.y)**2)
                    arrive_tick = math.floor(dist / fleet_speed)

                    if arrive_tick > needed_by_tick:
                        continue
                    
                else:
                    angle_np, arrive_tick = find_angle_to_moving_planet(p_np, p, sent_reinforcements, obs.angular_velocity)

                if angle_np is None or arrive_tick is None:
                    continue
                
                moves.append([p_np.id, angle_np, sent_reinforcements])
                exhausted_planets_id.add(p_np.id)
                reinforcement_trajectories.append({
                    "mine": p_np,
                    "target": p,
                    "angle": angle_np,
                    "total_ships": sent_reinforcements,
                    "arrive_tick": arrive_tick
                })
                break

        for m in sorted(lobs.get("mine", []), key=lambda p: p.ships, reverse=True):
            if m.id in exhausted_planets_id:
                continue

            if m.ships < MIN_SHIPS_MINE_ATTACK:
                continue

            candidate_targets = []
            for t in lobs.get("targets", []):
                if t.id in comet_planet_ids:
                    continue

                score = get_custom_score(m, t)
                candidate_targets.append((m, t, score))

            candidate_targets = sorted(candidate_targets, key=lambda x: x[2], reverse=True)
            
        
            for m, t, s in candidate_targets[:3]:
                m_available_ships = m.ships
        
                if m.id in under_attack:
                    enemy_ships = sum(
                        att["fleet"].ships
                        for att in under_attack[m.id]["fleets"]
                    )
                    m_available_ships = max(0, m.ships - enemy_ships)
        
                if m_available_ships < MIN_SHIPS_MINE_ATTACK:
                    continue
                
                nearest_planets = get_closest_planets_to_target(lobs.get("mine", []), t)
                safe_nearest_planets = []
                for p, dist in nearest_planets: # check which planets are fit to attack and are not vulnerable
                    if p.id == m.id or p.id in exhausted_planets_id:
                        continue
                    
                    available_ships = p.ships
                    
                    if p.id in under_attack:
                        enemy_ships = sum(
                            att["fleet"].ships 
                            for att in under_attack[p.id]["fleets"]
                        )
                        available_ships = max(0, p.ships - enemy_ships)
        
                    if available_ships < MIN_SHIPS_MINE_ATTACK:
                        continue
                    
                    safe_nearest_planets.append((p, dist, available_ships))
                
                owned_count = len(lobs.get("mine", []))
                total_count = len(lobs.get("planets", []))
        
                en_route = 0
                if fleet_trajectories:
                    en_route = sum(
                        f["total_ships"]
                        for f in fleet_trajectories
                        if f["target"].id == t.id
                    )
        
                needed_now = t.ships + 1
                if t.owner != -1:
                    needed_now += 3 * t.production
                
                if owned_count < total_count * 0.75: # release all havoc when targets less than ~25%
                    if en_route >= needed_now:
                        continue
                
                base_ships = max(MIN_SHIPS_MINE_ATTACK, needed_now - en_route)
                
                extra_ships = 0
                fleet_speed = 0
                angle = None
                arrive_tick = None
        
                if m_available_ships >= base_ships: # single attack
                    if t.id in moving_planets: # single moving planet
                        total_ships = base_ships
                        
                        for _ in range(3):
                            angle, arrive_tick = find_angle_to_moving_planet(m, t, total_ships, obs.angular_velocity)

                            if angle is None:
                                break

                            if t.owner != -1:
                                new_total_ships = base_ships + arrive_tick * t.production
                            else:
                                new_total_ships = base_ships

                            if new_total_ships > m_available_ships:
                                angle = None
                                break
                            
                            if new_total_ships == total_ships:
                                break

                            total_ships = new_total_ships
                        extra_ships = total_ships - base_ships
                            
                    else: # single static planet
                        angle = calculate_angle(m, t) # single static unowned
                        total_ships = base_ships
                        dist = math.sqrt((t.x - m.x)**2 + (t.y - m.y)**2)
                        fleet_speed = 1.0 + (MAX_SPEED - 1.0) * (math.log(max(1, total_ships)) / math.log(1000)) ** 1.5
                        arrive_tick = math.floor(dist / fleet_speed)
                        
                        if t.owner != -1: # single static owned
                            for _ in range(3):
                                fleet_speed = 1.0 + (MAX_SPEED - 1.0) * (math.log(max(1, total_ships)) / math.log(1000)) ** 1.5
                                turns_to_arrive = math.floor(dist / fleet_speed)
                                
                                extra_ships = turns_to_arrive * t.production
                                new_total_ships = base_ships + extra_ships

                                if new_total_ships > m_available_ships:
                                    angle = None
                                    arrive_tick = None
                                    break

                                arrive_tick = turns_to_arrive
                                
                                if new_total_ships == total_ships:
                                    break
                                
                                total_ships = new_total_ships

                            extra_ships = total_ships - base_ships
                            
                    if angle is not None and arrive_tick is not None:
                        fleet_speed = 1.0 + (MAX_SPEED - 1.0) * (math.log(max(1, total_ships)) / math.log(1000)) ** 1.5

                        collides_sun = sun_collision(m, fleet_speed, angle)
                        if collides_sun:
                            continue
                            
                        moves.append([m.id, angle, total_ships])
                        exhausted_planets_id.add(m.id)
                        fleet_trajectories.append({
                            "mine": m,
                            "target": t,
                            "angle": angle,
                            "total_ships": total_ships,
                            "arrive_tick": arrive_tick
                        })
                
                elif m_available_ships < base_ships and len(lobs.get("mine", [])) > 1 and t.ships >= MIN_SHIPS_TARGET_COOP_ATTACK: # coop attack
                    accum = m_available_ships
                    attacking_planets = [{"planet": m, "ships": m_available_ships}]
                    coop_sent = False
                    
                    for p, dist, p_available_ships in safe_nearest_planets:
                        if coop_sent:
                            break
                        
                        attacking_planets.append({"planet": p, "ships": p_available_ships})
                        accum += p_available_ships
        
                        if len(attacking_planets) > COOP_PLANET_CAP:
                            break
                        
                        if accum < base_ships:
                            continue
                            
                        if t.id not in moving_planets: # coop static planet
                            if t.owner == -1: # coop static unowned
                                remainder = base_ships
                                planned = []
                                for a_p in attacking_planets:
                                    p = a_p["planet"]
                                    p_ships = min(a_p["ships"], remainder)
                                    
                                    if p_ships > 0:
                                        p_ships = min(a_p["ships"], max(p_ships, MIN_SHIPS_MINE_ATTACK))
        
                                    if p_ships <= 0:
                                        continue
                                    
                                    angle = calculate_angle(p, t)
                                    dist = math.sqrt((p.x - t.x)**2 + (p.y - t.y)**2)
                                    fleet_speed = 1.0 + (MAX_SPEED - 1.0) * (math.log(p_ships) / math.log(1000)) ** 1.5
                                    arrive_tick = math.floor(dist / fleet_speed)
                                    
                                    collides_sun = sun_collision(p, fleet_speed=fleet_speed, angle=angle)
                                    if collides_sun:
                                        break
        
                                    remainder -= p_ships
                                        
                                    planned.append([p, angle, p_ships, arrive_tick])
        
                                if remainder > 0:
                                    continue
                                    
                                for move in planned:
                                    fleet_trajectories.append({
                                        "mine": move[0],
                                        "target": t,
                                        "angle": move[1],
                                        "total_ships": move[2],
                                        "arrive_tick": move[3]
                                    })
                                    exhausted_planets_id.add(move[0].id)
                                    move[0] = move[0].id
                                    moves.append(move)
        
                                coop_sent = True
                                break
                                    
                            else: # coop static owned
                                required_ships = calculate_req_ships(attacking_planets, t, base_ships)
                                remainder = required_ships
                                
                                if accum < required_ships: 
                                    continue
                                    
                                planned = []
                                for a_p in attacking_planets:
                                    p = a_p["planet"]
                                    p_ships = min(a_p["ships"], remainder)
                                    
                                    if p_ships > 0:
                                        p_ships = min(a_p["ships"], max(p_ships, MIN_SHIPS_MINE_ATTACK))
        
                                    if p_ships <= 0:
                                        continue
                                        
                                    angle = calculate_angle(p, t)
                                    dist = math.sqrt((p.x - t.x)**2 + (p.y - t.y)**2)
                                    fleet_speed = 1.0 + (MAX_SPEED - 1.0) * (math.log(p_ships) / math.log(1000)) ** 1.5
                                    arrive_tick = math.floor(dist / fleet_speed)
                                    
                                    collides_sun = sun_collision(p, fleet_speed=fleet_speed, angle=angle)
                                    if collides_sun:
                                        continue
        
                                    remainder -= p_ships
                                    
                                    planned.append([p, angle, p_ships, arrive_tick])
        
                                if remainder > 0:
                                    continue
                                
                                for move in planned:
                                    fleet_trajectories.append({
                                        "mine": move[0],
                                        "target": t,
                                        "angle": move[1],
                                        "total_ships": move[2],
                                        "arrive_tick": move[3]
                                    })
                                    exhausted_planets_id.add(move[0].id)
                                    move[0] = move[0].id
                                    moves.append(move)
        
                                coop_sent = True
                                break
                        
                        else: # coop moving planet
                            planet_trajectories = get_planet_trajectories(t, obs.angular_velocity)
                            if t.owner == -1: # coop moving unowned
                                remainder = base_ships
                                planned = []
                                for a_p in attacking_planets:
                                    p = a_p["planet"]
                                    p_ships = min(a_p["ships"], remainder)
                                    
                                    if p_ships > 0:
                                        p_ships = min(a_p["ships"], max(p_ships, MIN_SHIPS_MINE_ATTACK))
        
                                    if p_ships <= 0:
                                        continue
        
                                    angle, arrive_tick = find_angle_to_moving_planet(p, t, p_ships, obs.angular_velocity)
                                    
                                    if angle is None or arrive_tick is None:
                                        continue
                                        
                                    planned.append([p, angle, p_ships, arrive_tick])
                                    remainder -= p_ships
        
                                if remainder > 0:
                                    continue
        
                                for move in planned:
                                    fleet_trajectories.append({
                                        "mine": move[0],
                                        "target": t,
                                        "angle": move[1],
                                        "total_ships": move[2],
                                        "arrive_tick": move[3]
                                    })
                                    exhausted_planets_id.add(move[0].id)
                                    move[0] = move[0].id
                                    moves.append(move)
        
                                coop_sent = True
                                break
                        
                            else: # coop moving owned
                                required_ships = calculate_req_ships_moving(attacking_planets, t, base_ships, obs.angular_velocity)
                                remainder = required_ships
                                planned = []
        
                                if accum < required_ships:
                                    continue
                                
                                for a_p in attacking_planets:
                                    p = a_p["planet"]
                                    p_ships = min(a_p["ships"], remainder)
                                    
                                    if p_ships > 0:
                                        p_ships = min(a_p["ships"], max(p_ships, MIN_SHIPS_MINE_ATTACK))   
                                        
                                    if p_ships <= 0:
                                        continue
                                    
                                    fleet_speed = 1.0 + (MAX_SPEED - 1.0) * (math.log(max(1, p_ships)) / math.log(1000)) ** 1.5
        
                                    angle, arrive_tick = find_angle_to_moving_planet(p, t, p_ships, obs.angular_velocity)
        
                                    if angle is None or arrive_tick is None:
                                        continue
                                    
                                    remainder -= p_ships
        
                                    planned.append([p, angle, p_ships, arrive_tick])
        
                                if remainder > 0:
                                    continue
                                
                                for move in planned:
                                    fleet_trajectories.append({
                                        "mine": move[0],
                                        "target": t,
                                        "angle": move[1],
                                        "total_ships": move[2],
                                        "arrive_tick": move[3]
                                    })
                                    exhausted_planets_id.add(move[0].id)
                                    move[0] = move[0].id
                                    moves.append(move)
        
                                coop_sent = True
                                break
        
        return moves
    return agent

_ABC_AGENT = None

SUN_X, SUN_Y = 50.0, 50.0
SUN_RADIUS = 10.0
MAX_SPEED = 6.0
DECOY_THRESHOLD = 8

def fleet_speed(ships: int) -> float:
    """Tính tốc độ đội tàu dựa trên số ship: càng nhiều ship thì càng nhanh, nhưng tăng theo log."""
    if ships <= 0:
        return 1.0
    return 1.0 + (MAX_SPEED - 1.0) * (math.log(max(ships, 1)) / math.log(1000)) ** 1.5


def travel_time(x1: float, y1: float, x2: float, y2: float, ships: int) -> float:
    """Ước lượng thời gian bay từ điểm 1 tới điểm 2."""
    dist = math.hypot(x2 - x1, y2 - y1)
    return dist / fleet_speed(ships) if ships > 0 else 999.0


def line_seg_min_dist(x1: float, y1: float, x2: float, y2: float, px: float, py: float) -> float:
    """Khoảng cách ngắn nhất từ điểm (px, py) tới đoạn thẳng nối (x1, y1) -> (x2, y2)."""
    dx, dy = x2 - x1, y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq == 0:
        return math.hypot(x1 - px, y1 - py)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / len_sq))
    return math.hypot(x1 + t * dx - px, y1 + t * dy - py)


def path_crosses_sun(x1: float, y1: float, x2: float, y2: float, margin: float = 1.5) -> bool:
    """Kiểm tra đường bay có đi quá gần mặt trời không."""
    return line_seg_min_dist(x1, y1, x2, y2, SUN_X, SUN_Y) < SUN_RADIUS + margin


def predict_orbit(x: float, y: float, omega: float, dt: float):
    """Dự đoán vị trí tương lai của một hành tinh đang quay quanh mặt trời sau dt thời gian."""
    theta = math.atan2(y - SUN_Y, x - SUN_X)
    r = math.hypot(x - SUN_X, y - SUN_Y)
    return SUN_X + r * math.cos(theta + omega * dt), SUN_Y + r * math.sin(theta + omega * dt)


def solve_intercept(fx: float, fy: float, tx: float, ty: float, orbiting: bool, omega: float, ships: int, iterations: int = 25):
    """Tìm điểm cần bắn tới để gặp target đang di chuyển trên quỹ đạo."""
    if not orbiting:
        t = travel_time(fx, fy, tx, ty, ships)
        return tx, ty, t
    t = travel_time(fx, fy, tx, ty, ships)
    ix, iy = tx, ty
    for _ in range(iterations):
        ix, iy = predict_orbit(tx, ty, omega, t)
        t2 = travel_time(fx, fy, ix, iy, ships)
        if abs(t2 - t) < 0.05:
            break
        t = t2
    return ix, iy, t


def safe_angle(x1: float, y1: float, x2: float, y2: float) -> float:
    """Trả về góc bắn an toàn, tránh đi xuyên gần mặt trời nếu đường thẳng bị nguy hiểm."""
    direct = math.atan2(y2 - y1, x2 - x1)
    if not path_crosses_sun(x1, y1, x2, y2, margin=1.5):
        return direct
    d = math.hypot(x1 - SUN_X, y1 - SUN_Y)
    if d <= SUN_RADIUS + 1.0:
        return direct
    half = math.asin(min(1.0, (SUN_RADIUS + 1.0) / d))
    to_sun = math.atan2(SUN_Y - y1, SUN_X - x1)
    cw = to_sun + half
    ccw = to_sun - half
    def adiff(a):
        dd = (a - direct) % (2 * math.pi)
        return min(dd, 2 * math.pi - dd)
    return cw if adiff(cw) < adiff(ccw) else ccw


def is_decoy_fleet(fleet, planets, omega):
    """Đoán fleet địch có phải mồi nhử không, để không phản ứng quá mức."""
    if fleet['ships'] < DECOY_THRESHOLD:
        return True
    tgt_id = None
    best_dist = float('inf')
    for p in planets.values():
        d = math.hypot(fleet['x'] - p['x'], fleet['y'] - p['y'])
        if d < best_dist:
            best_dist = d
            tgt_id = p['id']
    if tgt_id is None:
        return True
    tgt = planets.get(tgt_id)
    if tgt is None:
        return True
    r = math.hypot(tgt['x'] - SUN_X, tgt['y'] - SUN_Y)
    is_orb = (r + tgt['radius']) < 48.0
    ships_needed = tgt['ships'] + 1
    if fleet['ships'] < ships_needed * 0.4:
        return True
    return False


def ships_needed_for_takeover(tgt_ships, tgt_prod, tt, owner, margin=1.05):
    """Tính số ship cần gửi để chiếm target sau khi target đã sản xuất thêm trong thời gian bay."""
    if owner == -1:
        return int(tgt_ships * margin) + 1
    growth = tgt_prod * tt
    return int((tgt_ships + growth) * margin) + 1


def planet_under_threat(p_id, fleets, planets, player, omega):
    """Ước lượng tổng ship địch đang có khả năng lao tới hành tinh p_id của mình."""
    incoming = 0
    for f in fleets.values():
        if f['owner'] == player:
            continue
        best_tgt, best_d = None, float('inf')
        for p in planets.values():
            if p['id'] == f['from']:
                continue
            d = math.hypot(f['x'] - p['x'], f['y'] - p['y'])
            if d < best_d:
                best_d = d
                best_tgt = p['id']
        if best_tgt == p_id:
            r = math.hypot(planets[p_id]['x'] - SUN_X, planets[p_id]['y'] - SUN_Y)
            is_orbiting = (r + planets[p_id]['radius']) < 48.0
            if is_orbiting:
                ix, iy = predict_orbit(planets[p_id]['x'], planets[p_id]['y'], omega, travel_time(f['x'], f['y'], planets[p_id]['x'], planets[p_id]['y'], int(f['ships'])))
                d = math.hypot(ix - planets[p_id]['x'], iy - planets[p_id]['y'])
            else:
                d = math.hypot(f['x'] - planets[p_id]['x'], f['y'] - planets[p_id]['y'])
            if d < 50:
                incoming += f['ships']
    return incoming


def compute_tangent_points(x1: float, y1: float, margin: float = 2.0):
    """Tính hai góc tiếp tuyến từ điểm hiện tại tới vòng nguy hiểm quanh mặt trời."""
    d = math.hypot(x1 - SUN_X, y1 - SUN_Y)
    if d <= SUN_RADIUS + margin:
        return None, None
    half_angle = math.asin(min(1.0, (SUN_RADIUS + margin) / d))
    to_sun = math.atan2(SUN_Y - y1, SUN_X - x1)
    return to_sun + half_angle, to_sun - half_angle


def multi_leg_path(x1: float, y1: float, x2: float, y2: float, margin: float = 2.0):
    """Chỉ dùng đường nhiều chặng nếu đường đi thẳng tới target bị mặt trời chắn."""
    if not path_crosses_sun(x1, y1, x2, y2, margin):
        return [(x2, y2)], math.hypot(x2 - x1, y2 - y1)
    beacon_ring = SUN_RADIUS + 15.0
    waypoints = []
    for angle in [0, math.pi/2, math.pi, 3*math.pi/2]:
        bx = SUN_X + beacon_ring * math.cos(angle)
        by = SUN_Y + beacon_ring * math.sin(angle)
        if not path_crosses_sun(x1, y1, bx, by, margin) and not path_crosses_sun(bx, by, x2, y2, margin):
            waypoints.append((bx, by))
    if not waypoints:
        return None, float('inf')
    best_wp = None
    best_dist = float('inf')
    for wx, wy in waypoints:
        d = math.hypot(wx - x1, wy - y1) + math.hypot(x2 - wx, y2 - wy)
        if d < best_dist:
            best_dist = d
            best_wp = (wx, wy)
    if best_wp:
        return [best_wp, (x2, y2)], best_dist
    
    return None, float('inf')

def estimate_capture_bonus(src_x: float, src_y: float, planet, omega: float, ships: int) -> float:
    """Trả về điểm thưởng cho các hành tinh có khoảng thời gian chiếm dễ hơn."""
    r = math.hypot(planet['x'] - SUN_X, planet['y'] - SUN_Y)
    if (r + planet['radius']) >= 48.0:
        return 0.0
    if not path_crosses_sun(src_x, src_y, planet['x'], planet['y'], margin=2.0):
        return 3.0
    safe_count = 0
    for offset in range(-6, 7):
        fx, fy = predict_orbit(planet['x'], planet['y'], omega, offset)
        if not path_crosses_sun(src_x, src_y, fx, fy, margin=2.0):
            safe_count += 1
    return (safe_count / 13.0) * 5.0

def main_agent(obs):
    """
    Hàm chính Kaggle gọi mỗi turn.

    Input obs chứa trạng thái hiện tại của game:
    - player: id của mình, từ 0 đến 3.

    - planets: danh sách tất cả hành tinh, bao gồm cả comet.
    Mỗi phần tử có dạng:
        [id, owner, x, y, radius, ships, production]
    Trong đó:
    + id: mã hành tinh.
    + owner: người sở hữu hành tinh.
    + x, y: tọa độ tâm hành tinh.
    + radius: bán kính hành tinh.
    + ships: số tàu hiện có trên hành tinh.
    + production: tốc độ sinh tàu.

    - fleets: danh sách tất cả fleet đang bay.
    Mỗi phần tử có dạng:
        [id, owner, x, y, angle, from_planet_id, ships]
    Trong đó:
    + angle: góc bay hiện tại, tính bằng radian.
    + from_planet_id: hành tinh xuất phát.

    - angular_velocity: tốc độ quay của hành tinh quanh mặt trời, tính bằng radian/turn.

    - initial_planets: danh sách vị trí hành tinh lúc bắt đầu game.
    Dạng giống planets:
        [id, owner, x, y, radius, ships, production]

    - comets: dữ liệu nhóm comet đang hoạt động.
    Mỗi phần tử có dạng:
        {planet_ids, paths, path_index}

    - comet_planet_ids: danh sách id các hành tinh là comet.

    - remainingOverageTime: thời gian overage còn lại, tính bằng giây.

    Output là moves:
    - Mỗi move có dạng:
    [src_id, angle, send]
    Trong đó:
    + src_id: hành tinh của mình sẽ gửi quân.
    + angle: góc bắn/gửi fleet, tính bằng radian.
    + send: số ship gửi đi.
    """
    if isinstance(obs, dict):
        player = obs.get('player', 0)
        planets_data = obs.get('planets', [])
        fleets_data = obs.get('fleets', [])
        step = obs.get('step', 0)
        omega = obs.get('angular_velocity', 0.03)
    else:
        player = getattr(obs, 'player', 0)
        planets_data = getattr(obs, 'planets', [])
        fleets_data = getattr(obs, 'fleets', [])
        step = getattr(obs, 'step', 0)
        omega = getattr(obs, 'angular_velocity', 0.03)

    planets = {}
    for p in planets_data:
        pid, owner, x, y, radius, ships, prod = p[:7]
        r = math.hypot(x - SUN_X, y - SUN_Y)
        planets[pid] = {
            'id': pid, 'owner': owner, 'x': x, 'y': y,
            'radius': radius, 'ships': float(ships), 'prod': float(prod),
            'is_orb': (r + radius) < 48.0
        }

    fleets = {}
    for f in fleets_data:
        fleets[f[0]] = {
            'id': f[0], 'owner': f[1], 'x': f[2], 'y': f[3],
            'angle': f[4], 'from': f[5], 'ships': float(f[6])
        }

    my = [p for p in planets.values() if p['owner'] == player]
    if not my:
        return []

    enemy = [p for p in planets.values() if p['owner'] != player and p['owner'] != -1]
    neutrals = [p for p in planets.values() if p['owner'] == -1]

    my_prod = sum(p['prod'] for p in my)
    my_ships = sum(p['ships'] for p in my)
    enemy_prod = sum(p['prod'] for p in enemy) if enemy else 0
    enemy_ships = sum(p['ships'] for p in enemy) if enemy else 0
    prod_ratio = my_prod / enemy_prod if enemy_prod > 0 else 999
    ship_ratio = my_ships / enemy_ships if enemy_ships > 0 else 999

    my_planet_count = len(my)
    neighbor_count = sum(1 for t in neutrals if any(math.hypot(t['x'] - p['x'], t['y'] - p['y']) < 35 for p in my))

    nearby_larger_planets = []
    for src in my:
        for t in (neutrals + enemy):
            d = math.hypot(t['x'] - src['x'], t['y'] - src['y'])
            if d < 40 and t['prod'] >= src['prod'] * 0.8 and t['radius'] >= src['radius'] * 0.8:
                nearby_larger_planets.append((src['id'], t['id'], d))

    real_enemy_fleets = {f_id: f for f_id, f in fleets.items() if f['owner'] != player and not is_decoy_fleet(f, planets, omega)}

    in_flight_from = set()
    in_flight_to = set()
    for f in fleets.values():
        if f['owner'] == player and f['from'] is not None:
            in_flight_from.add(f['from'])
            best_tgt, best_d = None, float('inf')
            for p in planets.values():
                if p['id'] == f['from']:
                    continue
                d = math.hypot(f['x'] - p['x'], f['y'] - p['y'])
                if d < best_d:
                    best_d = d
                    best_tgt = p['id']
            if best_tgt:
                in_flight_to.add(best_tgt)

    threats = {}
    for p in planets.values():
        if p['owner'] == player:
            threats[p['id']] = planet_under_threat(p['id'], fleets, planets, player, omega)

    smash_targets = set()
    for e in enemy:
        nearby_my_ships = sum(p['ships'] for p in my if math.hypot(p['x'] - e['x'], p['y'] - e['y']) < 50)
        if nearby_my_ships > e['ships'] * 0.95:
            smash_targets.add(e['id'])

    if smash_targets:
        phase = 'smash'
    elif my_ships > 120 and my_planet_count < 4 and enemy:
        phase = 'rush'
    elif my_planet_count < 3 or (neighbor_count > 0 and my_planet_count < 5):
        phase = 'expand'
    elif threats and any(t > my_ships * 0.25 for t in threats.values()):
        phase = 'counter_attack'
    elif prod_ratio > 4.0 and my_ships > 80 and my_planet_count >= 3:
        phase = 'crush'
    elif prod_ratio > 2.0 or ship_ratio > 2.5:
        phase = 'aggressive'
    elif my_prod < enemy_prod * 0.7:
        phase = 'defend'
    elif len(enemy) > 0 and len(my) >= 3 and my_prod > enemy_prod * 1.0:
        phase = 'dominate'
    else:
        phase = 'grow'

    moves = []

    targeted_this_turn = set()

    for src in my:
        if src['id'] in in_flight_from:
            handoff_relaunch = (
                ABC_SWITCH_STEP <= step <= 95
                and src['ships'] > 70
                and threats.get(src['id'], 0) < src['ships'] * 0.2
                and my_prod <= max(enemy_prod * 1.2, enemy_prod + 4)
            )
            if not handoff_relaunch:
                continue

        if src['ships'] < 10:
            continue

        if phase == 'expand':
            nearby_larger = {nl[1] for nl in nearby_larger_planets if nl[0] == src['id']}
            best_target = None
            best_score = -1e9
            for t in neutrals:
                if t['id'] == src['id']:
                    continue
                if t['id'] in in_flight_to or t['id'] in targeted_this_turn:
                    continue
                d = math.hypot(t['x'] - src['x'], t['y'] - src['y'])
                score = -d * 3 + t['prod'] * 3
                if nearby_larger and t['radius'] < src['radius'] * 0.7 and d > 25:
                    score -= 50
                if score > best_score:
                    best_score = score
                    best_target = t
            if best_target:
                r = math.hypot(best_target['x'] - SUN_X, best_target['y'] - SUN_Y)
                is_orbiting = (r + best_target['radius']) < 48.0
                ix, iy, tt = solve_intercept(src['x'], src['y'], best_target['x'], best_target['y'], is_orbiting, omega, int(src['ships']))
                if not path_crosses_sun(src['x'], src['y'], ix, iy, margin=1.5):
                    send = ships_needed_for_takeover(best_target['ships'], best_target['prod'], tt, best_target['owner'])
                    if src['ships'] >= send:
                        angle = safe_angle(src['x'], src['y'], ix, iy)
                        moves.append([src['id'], angle, send])
                        targeted_this_turn.add(best_target['id'])
                        src['ships'] -= send
                        if src['ships'] < 5:
                            break
            elif src['ships'] > 40:
                decoy_tgt = None
                decoy_score = -1e9
                for t in (enemy + neutrals):
                    if t['id'] == src['id']:
                        continue
                    if t['id'] in targeted_this_turn:
                        continue
                    d = math.hypot(t['x'] - src['x'], t['y'] - src['y'])
                    score = -d + (t['prod'] if t['owner'] != -1 else 0) * 5
                    if nearby_larger and t['radius'] < src['radius'] * 0.7 and d > 25:
                        score -= 50
                    if score > decoy_score:
                        decoy_score = score
                        decoy_tgt = t
                if decoy_tgt and src['ships'] > 25:
                    send = min(8, int(src['ships'] * 0.15))
                    if send >= 5:
                        r = math.hypot(decoy_tgt['x'] - SUN_X, decoy_tgt['y'] - SUN_Y)
                        is_orbiting = (r + decoy_tgt['radius']) < 48.0
                        ix, iy, tt = solve_intercept(src['x'], src['y'], decoy_tgt['x'], decoy_tgt['y'], is_orbiting, omega, int(src['ships']))
                        if not path_crosses_sun(src['x'], src['y'], ix, iy, margin=1.5):
                            angle = safe_angle(src['x'], src['y'], ix, iy)
                            moves.append([src['id'], angle, send])
                            targeted_this_turn.add(decoy_tgt['id'])
                            src['ships'] -= send
                            if src['ships'] < 10:
                                break

        need_defense = threats.get(src['id'], 0) > src['ships'] * 0.3

        if need_defense and phase != 'counter_attack':
            continue

        if need_defense and phase == 'counter_attack' and threats.get(src['id'], 0) >= src['ships'] * 0.5:
            continue

        if phase == 'counter_attack':
            best_enemy = None
            best_score = -1e9
            for t in enemy:
                if t['id'] in targeted_this_turn:
                    continue
                d = math.hypot(t['x'] - src['x'], t['y'] - src['y'])
                score = t['ships'] * 0.8 + t['prod'] * 8 - d
                if t['id'] in smash_targets:
                    score += 50
                if score > best_score:
                    best_score = score
                    best_enemy = t
            if best_enemy:
                r = math.hypot(best_enemy['x'] - SUN_X, best_enemy['y'] - SUN_Y)
                is_orbiting = (r + best_enemy['radius']) < 48.0
                ix, iy, tt = solve_intercept(src['x'], src['y'], best_enemy['x'], best_enemy['y'], is_orbiting, omega, int(src['ships']))
                if not path_crosses_sun(src['x'], src['y'], ix, iy, margin=1.5):
                    send = int(src['ships'] * 0.8)
                    send = max(send, int(best_enemy['ships'] * 1.1))
                    send = min(send, int(src['ships'] * 0.95))
                    if src['ships'] > send + 3:
                        angle = safe_angle(src['x'], src['y'], ix, iy)
                        moves.append([src['id'], angle, send])
                        targeted_this_turn.add(best_enemy['id'])
                        src['ships'] -= send

        best_tgt = None
        best_score = -1e9

        if phase == 'smash':
            candidates = [t for t in enemy if t['id'] in smash_targets]
        elif phase == 'rush':
            candidates = enemy
        elif phase == 'expand' or phase == 'opportunistic' or phase == 'aggressive' or phase == 'dominate':
            candidates = neutrals if phase not in ('aggressive', 'dominate') else (enemy + neutrals)
        elif phase == 'grow':
            candidates = [t for t in neutrals if threats.get(t['id'], 0) == 0]
        else:
            candidates = []

        for t in candidates:
            if t['id'] == src['id']:
                continue
            if t['id'] in in_flight_to:
                continue
            if t['id'] in targeted_this_turn:
                continue

            incoming = threats.get(t['id'], 0)
            if incoming > 0:
                continue

            r = math.hypot(t['x'] - SUN_X, t['y'] - SUN_Y)
            is_orbiting = t['is_orb']

            ix, iy, tt = solve_intercept(src['x'], src['y'], t['x'], t['y'], is_orbiting, omega, int(src['ships']))

            if path_crosses_sun(src['x'], src['y'], ix, iy, margin=1.5):
                waypoints, _ = multi_leg_path(src['x'], src['y'], ix, iy)
                if waypoints is None:
                    continue
                final_x, final_y = waypoints[-1]
                if path_crosses_sun(src['x'], src['y'], final_x, final_y, margin=1.5):
                    continue

            if is_orbiting:
                planet_future = predict_orbit(t['x'], t['y'], omega, tt)
                to_planet = math.atan2(planet_future[1] - src['y'], planet_future[0] - src['x'])
                to_target = math.atan2(t['y'] - src['y'], t['x'] - src['x'])
                diff = abs((to_planet - to_target) % (2 * math.pi))
                if diff > 0.5 and diff < (2 * math.pi - 0.5):
                    continue

            score = t['prod'] * 18 - tt * 2.5

            if t['owner'] == -1:
                score += 25

            if phase == 'aggressive' and t['owner'] != -1:
                score += 35 - t['ships'] * 0.12

            if phase == 'dominate' and t['owner'] != -1:
                score += 45 - t['ships'] * 0.08

            if phase == 'dominate' and t['owner'] == -1:
                score += 20

            if is_orbiting:
                score -= 6

            if src['ships'] > 50 and t['owner'] == -1:
                score += 12

            if src['prod'] > t['prod'] * 0.7:
                score += 8

            score += estimate_capture_bonus(src['x'], src['y'], t, omega, int(src['ships']))

            if score > best_score:
                best_score = score
                best_tgt = (t, ix, iy, tt)

        if best_tgt is None:
            continue

        tgt, ix, iy, tt = best_tgt

        if phase == 'smash':
            send = int(src['ships'] * 0.9)
            send = max(send, ships_needed_for_takeover(tgt['ships'], tgt['prod'], tt, tgt['owner']))
        elif phase == 'rush':
            send = int(src['ships'] * 0.8)
        elif phase == 'aggressive':
            send = int(src['ships'] * 0.4)
            send = max(send, ships_needed_for_takeover(tgt['ships'], tgt['prod'], tt, tgt['owner']))
            send = min(send, int(src['ships'] * 0.7))
        elif phase == 'dominate':
            send = int(src['ships'] * 0.5)
            send = max(send, ships_needed_for_takeover(tgt['ships'], tgt['prod'], tt, tgt['owner']))
            send = min(send, int(src['ships'] * 0.8))
        elif phase == 'opportunistic':
            send = ships_needed_for_takeover(tgt['ships'], tgt['prod'], tt, tgt['owner'])
            send = min(send, int(src['ships'] * 0.5))
        else:
            send = ships_needed_for_takeover(tgt['ships'], tgt['prod'], tt, tgt['owner'])

        if src['ships'] < send:
            continue

        angle = safe_angle(src['x'], src['y'], ix, iy)
        moves.append([src['id'], angle, send])
        targeted_this_turn.add(tgt['id'])

    if phase == 'expand':
        for src in my:
            if src['id'] in in_flight_from:
                continue
            if src['ships'] < 10:
                continue
            nearby_larger = [nl for nl in nearby_larger_planets if nl[0] == src['id']]
            if not nearby_larger:
                continue
            candidates = [t for t in (neutrals + enemy)
                          if t['id'] not in targeted_this_turn
                          and t['id'] not in in_flight_to
                          and t['owner'] != player]
            if not candidates:
                continue
            best_tgt = None
            best_score = -1e9
            for t in candidates:
                d = math.hypot(t['x'] - src['x'], t['y'] - src['y'])
                if d > 40:
                    continue
                score = t['prod'] * 5 - d
                if t['radius'] >= src['radius'] * 0.8 and t['prod'] >= src['prod'] * 0.8:
                    score += 40
                if score > best_score:
                    best_score = score
                    best_tgt = t
            if best_tgt:
                r = math.hypot(best_tgt['x'] - SUN_X, best_tgt['y'] - SUN_Y)
                is_orbiting = (r + best_tgt['radius']) < 48.0
                ix, iy, tt = solve_intercept(src['x'], src['y'], best_tgt['x'], best_tgt['y'], is_orbiting, omega, int(src['ships']))
                if not path_crosses_sun(src['x'], src['y'], ix, iy, margin=1.5):
                    send = ships_needed_for_takeover(best_tgt['ships'], best_tgt['prod'], tt, best_tgt['owner'])
                    if src['ships'] >= send:
                        angle = safe_angle(src['x'], src['y'], ix, iy)
                        moves.append([src['id'], angle, send])
                        targeted_this_turn.add(best_tgt['id'])
                        src['ships'] -= send

    return moves


def _obs_step(obs):
    if isinstance(obs, dict):
        return obs.get('step', 0)
    return getattr(obs, 'step', 0)


def _get_abc_agent():
    global _ABC_AGENT
    if _ABC_AGENT is not None:
        return _ABC_AGENT
    try:
        _ABC_AGENT = _make_abc_agent()
        return _ABC_AGENT
    except Exception:
        return None


def _reset_abc_agent():
    global _ABC_AGENT
    _ABC_AGENT = None


def _sanitize_external_moves(obs, moves):
    return moves or []


def _abc_should_open(obs):
    step = _obs_step(obs)
    if step >= ABC_SWITCH_STEP:
        return False
    if step < 18:
        return True
    if isinstance(obs, dict):
        player = obs.get('player', 0)
        planets_data = obs.get('planets', [])
        fleets_data = obs.get('fleets', [])
    else:
        player = getattr(obs, 'player', 0)
        planets_data = getattr(obs, 'planets', [])
        fleets_data = getattr(obs, 'fleets', [])

    my_planets = [p for p in planets_data if p[1] == player]
    if len(my_planets) < 2:
        return True

    my_prod = sum(float(p[6]) for p in my_planets)
    my_ground = sum(float(p[5]) for p in my_planets)
    friendly_air = sum(float(f[6]) for f in fleets_data if f[1] == player)
    enemy_prod_by_owner = {}
    for p in planets_data:
        if p[1] not in (player, -1):
            enemy_prod_by_owner[p[1]] = enemy_prod_by_owner.get(p[1], 0.0) + float(p[6])
    strongest_enemy_prod = max(enemy_prod_by_owner.values(), default=0.0)

    if len(my_planets) >= 4 and my_prod >= strongest_enemy_prod * 0.9:
        return False
    if my_ground < max(18.0, friendly_air * 0.45) and len(my_planets) >= 3:
        return False
    if step >= 42 and my_prod < strongest_enemy_prod * 0.75 and my_ground < 55:
        return False
    return True


def agent(obs):
    step = _obs_step(obs)
    if step <= 1:
        _reset_abc_agent()
    abc_agent = _get_abc_agent()
    if abc_agent is not None and _abc_should_open(obs):
        try:
            return _sanitize_external_moves(obs, abc_agent(obs))
        except Exception:
            return main_agent(obs)
    return main_agent(obs)


if __name__ == '__main__':
    print("v49c Minimal Strategic Enhancement loaded!")

# %% [code]
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
# Tắt cơ chế "envelope" của Kaggle để agent trả về list move trực tiếp.
os.environ['KAGGLE_ENVELOPES'] = '0'

import math

SUN_X, SUN_Y = 50.0, 50.0
SUN_RADIUS = 10.0
MAX_SPEED = 6.0
DECOY_THRESHOLD = 8

# Trong game này bản đồ gần như là hệ tọa độ 0..100.
# Mặt trời nằm giữa bản đồ tại (50, 50), bán kính 10.
# Các đường bay quá gần mặt trời thường nguy hiểm, nên nhiều hàm bên dưới
# đều kiểm tra "path_crosses_sun" trước khi quyết định gửi quân.


def fleet_speed(ships: int) -> float:
    """Tính tốc độ đội tàu dựa trên số ship: càng nhiều ship thì càng nhanh, nhưng tăng theo log."""
    if ships <= 0:
        return 1.0
    # Công thức này không tuyến tính. Nghĩa là 100 ship không nhanh gấp 100 lần 1 ship.
    # math.log làm tốc độ tăng chậm dần khi ship tăng nhiều.
    # MAX_SPEED là trần tốc độ mong muốn, còn 1.0 là tốc độ tối thiểu.
    # log(ships) / log(1000) chuẩn hóa mốc 1000 ship về khoảng gần 1.
    # Lũy thừa 1.5 làm đội ít ship tăng tốc chậm hơn, đội lớn hưởng lợi rõ hơn.
    return 1.0 + (MAX_SPEED - 1.0) * (math.log(max(ships, 1)) / math.log(1000)) ** 1.5


def travel_time(x1: float, y1: float, x2: float, y2: float, ships: int) -> float:
    """Ước lượng thời gian bay từ điểm 1 tới điểm 2."""
    dist = math.hypot(x2 - x1, y2 - y1)
    # Nếu ships <= 0 thì coi như không thể bay, trả số rất lớn để target này bị loại.
    return dist / fleet_speed(ships) if ships > 0 else 999.0


def line_seg_min_dist(x1: float, y1: float, x2: float, y2: float, px: float, py: float) -> float:
    # Tìm khoảng cách ngắn nhất từ điểm P(px, py) tới đoạn thẳng nối A(x1, y1) -> B(x2, y2).
    """Khoảng cách ngắn nhất từ điểm (px, py) tới đoạn thẳng nối (x1, y1) -> (x2, y2)."""
    # Đây là hàm toán học quan trọng để biết đường bay có chạm vùng mặt trời không.
    dx, dy = x2 - x1, y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq == 0:
        # Nếu điểm đầu và điểm cuối trùng nhau thì đoạn thẳng bị co lại thành một điểm.
        return math.hypot(x1 - px, y1 - py)
    # Chiếu điểm P lên đường thẳng AB. t bị kẹp trong [0, 1] để nằm trên đoạn AB, không vượt ra ngoài.
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / len_sq))
    return math.hypot(x1 + t * dx - px, y1 + t * dy - py)


def path_crosses_sun(x1: float, y1: float, x2: float, y2: float, margin: float = 1.5) -> bool:
    """Kiểm tra đường bay có đi quá gần mặt trời không."""
    return line_seg_min_dist(x1, y1, x2, y2, SUN_X, SUN_Y) < SUN_RADIUS + margin # ??? có cần thiết thêm margin vì nó chuyển động thẳng không?


def predict_orbit(x: float, y: float, omega: float, dt: float):
    # x, y là tâm hiện tại của hành tinh đang quay quanh mặt trời.
    """Dự đoán vị trí tương lai của một hành tinh đang quay quanh mặt trời sau dt thời gian."""
    theta = math.atan2(y - SUN_Y, x - SUN_X) # Góc lệch của hành tinh so vs trục Ox từ mặt trời.
    r = math.hypot(x - SUN_X, y - SUN_Y)
    # Giữ nguyên bán kính quỹ đạo r, chỉ cộng thêm góc quay omega * dt.
    return SUN_X + r * math.cos(theta + omega * dt), SUN_Y + r * math.sin(theta + omega * dt)


def solve_intercept(fx: float, fy: float, tx: float, ty: float, orbiting: bool, omega: float, ships: int, iterations: int = 25):
    """Tìm điểm cần bắn tới để gặp target đang di chuyển trên quỹ đạo."""
    if not orbiting:
        # Target đứng yên thì không cần bắn đón, cứ bắn vào tọa độ hiện tại.
        t = travel_time(fx, fy, tx, ty, ships)
        return tx, ty, t
    # Bắt đầu bằng thời gian bay tới vị trí hiện tại, rồi lặp để sửa điểm đón.
    # Ví dụ:
    # - Lần 1 đoán bay mất 8 giây tới vị trí hiện tại.
    # - Sau 8 giây target đã quay sang vị trí khác.
    # - Tính lại thời gian bay tới vị trí mới đó.
    # - Lặp đến khi thời gian cũ và mới gần giống nhau.
    t = travel_time(fx, fy, tx, ty, ships)
    ix, iy = tx, ty
    for _ in range(iterations):
        # Dự đoán target sẽ ở đâu sau t thời gian, rồi tính lại thời gian bay tới điểm đó.
        ix, iy = predict_orbit(tx, ty, omega, t)
        t2 = travel_time(fx, fy, ix, iy, ships)
        if abs(t2 - t) < 0.05: # @ Nếu thời gian bay đã hội tụ ổn định, có thể cần chỉnh thêm
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
    # half là góc lệch tối thiểu để đường bay tiếp tuyến với vùng cấm quanh mặt trời.
    half = math.asin(min(1.0, (SUN_RADIUS + 1.0) / d)) # half = sin(R/khoảng cách) trong tam giác vuông tạo bởi tiếp tuyến.
    to_sun = math.atan2(SUN_Y - y1, SUN_X - x1)
    cw = to_sun + half
    ccw = to_sun - half
    def adiff(a):
        # Độ lệch góc nhỏ nhất giữa góc a và góc direct, có xử lý vòng 2*pi.
        dd = (a - direct) % (2 * math.pi)
        return min(dd, 2 * math.pi - dd)
    # Chọn hướng né mặt trời mà lệch ít nhất so với đường bắn trực tiếp.
    return cw if adiff(cw) < adiff(ccw) else ccw


def is_decoy_fleet(fleet, planets, omega):
    # @ có thể thiết kế thằng này để nhận diện các fleet địch có khả năng là mồi nhử, nhằm tránh phản ứng quá mức và bị dụ điên cuồng vào bẫy.
    """Đoán fleet địch có phải mồi nhử không, để không phản ứng quá mức."""
    if fleet['ships'] < DECOY_THRESHOLD:
        # Fleet quá nhỏ thường chỉ là mồi nhử hoặc không đủ chiếm hành tinh.
        return True
    tgt_id = None
    best_dist = float('inf')
    # Vì dữ liệu fleet không nói rõ target, code đoán target là hành tinh gần fleet nhất.
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
    # Biến is_orb đang được tính nhưng không dùng ở bản này.
    # Có thể tác giả từng định xử lý riêng target đang orbit, nhưng sau đó bỏ dở.
    # Nếu số ship quá thấp so với lực phòng thủ target thì coi là mồi nhử/yếu.
    if fleet['ships'] < ships_needed * 0.4:
        return True
    return False


def ships_needed_for_takeover(tgt_ships, tgt_prod, tt, owner, margin=1.05):
    """Tính số ship cần gửi để chiếm target sau khi target đã sản xuất thêm trong thời gian bay."""
    if owner == -1:
        # Hành tinh trung lập không sinh thêm theo owner địch, chỉ cần hơn số ship hiện tại một chút.
        return int(tgt_ships * margin) + 1
    growth = tgt_prod * tt
    return int((tgt_ships + growth) * margin) + 1


def planet_under_threat(p_id, fleets, planets, player, omega):
    # @ Có thể cải tiến cái này bằng cách ước lượng thời gian tới của mỗi fleet, rồi so sánh với thời gian sinh thêm ship của hành tinh để có cái nhìn chính xác hơn về mức độ đe dọa.
    """Ước lượng tổng ship địch đang có khả năng lao tới hành tinh p_id của mình."""
    incoming = 0
    for f in fleets.values():
        if f['owner'] == player:
            continue
        best_tgt, best_d = None, float('inf')
        # Đoán target của fleet bằng hành tinh gần fleet nhất, bỏ qua hành tinh xuất phát.
        for p in planets.values():
            if p['id'] == f['from']:
                continue
            d = math.hypot(f['x'] - p['x'], f['y'] - p['y'])
            if d < best_d:
                best_d = d
                best_tgt = p['id']
        if best_tgt == p_id:
            r = math.hypot(planets[p_id]['x'] - SUN_X, planets[p_id]['y'] - SUN_Y)
            is_orbiting = (r + planets[p_id]['radius']) < 48.0 # @ Quy ước này có thể cần sửa lại
            if is_orbiting:
                # Nếu hành tinh đang quay, so fleet với vị trí tương lai của hành tinh.
                ix, iy = predict_orbit(planets[p_id]['x'], planets[p_id]['y'], omega, travel_time(f['x'], f['y'], planets[p_id]['x'], planets[p_id]['y'], int(f['ships'])))
                d = math.hypot(ix - planets[p_id]['x'], iy - planets[p_id]['y'])
            else:
                d = math.hypot(f['x'] - planets[p_id]['x'], f['y'] - planets[p_id]['y'])
            # @ Ngưỡng 50 là heuristic: đủ gần thì tính là mối đe dọa.
            if d < 50:
                incoming += f['ships']
    return incoming


# =============================================================================
# MULTI-LEG PATH PLANNER (minimal - just for hard targets), Bộ lập kế hoạch đi nhiều chặng
# (bản đơn giản, chỉ dùng cho các mục tiêu khó tiếp cận), A → hành tinh trung gian → B
# =============================================================================

def compute_tangent_points(x1: float, y1: float, margin: float = 2.0): # @ ko được dùng
    """Tính hai góc tiếp tuyến từ điểm hiện tại tới vòng nguy hiểm quanh mặt trời."""
    
    # Khoảng cách từ điểm hiện tại tới tâm mặt trời.
    d = math.hypot(x1 - SUN_X, y1 - SUN_Y)
    
    # Nếu đang nằm trong hoặc quá sát vùng nguy hiểm thì không tính được tiếp tuyến an toàn.
    if d <= SUN_RADIUS + margin:
        return None, None
    
    # Bán kính vùng nguy hiểm = bán kính mặt trời + vùng đệm an toàn.
    # half_angle là góc lệch từ hướng nhìn thẳng vào mặt trời tới đường tiếp tuyến.
    half_angle = math.asin(min(1.0, (SUN_RADIUS + margin) / d))
    
    # Góc từ điểm hiện tại nhìn thẳng tới tâm mặt trời.
    to_sun = math.atan2(SUN_Y - y1, SUN_X - x1)
    
    # Trả về 2 hướng tiếp tuyến: lệch sang 2 bên của mặt trời.
    return to_sun + half_angle, to_sun - half_angle


def multi_leg_path(x1: float, y1: float, x2: float, y2: float, margin: float = 2.0):
    """Chỉ dùng đường nhiều chặng nếu đường đi thẳng tới target bị mặt trời chắn."""
    
    # Nếu đi thẳng từ điểm hiện tại tới target không cắt mặt trời thì đi thẳng luôn.
    if not path_crosses_sun(x1, y1, x2, y2, margin):
        return [(x2, y2)], math.hypot(x2 - x1, y2 - y1)
    
    # Tạo một vòng điểm trung gian quanh mặt trời.
    # Các điểm này nằm cách tâm mặt trời một khoảng SUN_RADIUS + 15.
    beacon_ring = SUN_RADIUS + 15.0 # @ Khoảng cách này là heuristic: đủ xa để tránh bị cắt nhưng không quá xa để không bị phạt đường bay dài.
    waypoints = []
    
    # Thử 4 điểm quanh mặt trời: phải, trên, trái, dưới, hình thoi bao quanh hình tròn
    # @ có nên dùng giao của hai tiếp tuyến, sau đó tìm planet gần nhất với giao điểm đó để làm waypoint, planet đó phải ở phía ko phải mặt trời.
    for angle in [0, math.pi/2, math.pi, 3*math.pi/2]: # @ Có thể tăng số điểm này lên để tìm được đường vòng tốt hơn, nhưng sẽ tốn thời gian tính toán hơn.
        bx = SUN_X + beacon_ring * math.cos(angle)
        by = SUN_Y + beacon_ring * math.sin(angle)
        
        # Một waypoint hợp lệ nếu cả 2 chặng đều an toàn:
        # điểm hiện tại -> waypoint
        # waypoint -> target
        if not path_crosses_sun(x1, y1, bx, by, margin) and not path_crosses_sun(bx, by, x2, y2, margin):
            waypoints.append((bx, by))
    
    # Nếu không có điểm trung gian nào an toàn thì coi như không tìm được đường vòng.
    if not waypoints:
        return None, float('inf')
    
    # Chọn waypoint làm tổng quãng đường ngắn nhất.
    best_wp = None
    best_dist = float('inf')
    
    for wx, wy in waypoints:
        # Tổng khoảng cách = chặng 1 + chặng 2.
        d = math.hypot(wx - x1, wy - y1) + math.hypot(x2 - wx, y2 - wy)
        
        if d < best_dist:
            best_dist = d
            best_wp = (wx, wy)
    
    # Nếu tìm được waypoint tốt nhất thì trả về đường đi 2 chặng:
    # hiện tại -> waypoint -> target.
    if best_wp:
        return [best_wp, (x2, y2)], best_dist
    
    return None, float('inf')


# =============================================================================
# @ Ước lượng thời điểm/cơ hội thuận lợi để chiếm hành tinh (simplified - just scoring bonus)
# =============================================================================

def estimate_capture_bonus(src_x: float, src_y: float, planet, omega: float, ships: int) -> float:
    """Trả về điểm thưởng cho các hành tinh có khoảng thời gian chiếm dễ hơn."""
    
    # Tính khoảng cách từ hành tinh tới mặt trời.
    r = math.hypot(planet['x'] - SUN_X, planet['y'] - SUN_Y)
    
    # Nếu hành tinh không quay quanh mặt trời thì không có bonus đặc biệt.
    if (r + planet['radius']) >= 48.0: # @
        return 0.0
    
    # Kiểm tra xem bắn thẳng hiện tại có an toàn không.
    # Nếu không cắt gần mặt trời thì coi như rất dễ chiếm.
    if not path_crosses_sun(src_x, src_y, planet['x'], planet['y'], margin=2.0):
        return 3.0
    
    # Kiểm tra nhiều vị trí tương lai/quá khứ gần đó.
    # Càng nhiều vị trí có thể bắn an toàn thì càng có "cửa sổ chiếm" rộng.
    safe_count = 0
    
    for offset in range(-6, 7): # @ Kiểm tra vị trí của hành tinh trong khoảng thời gian từ -6 tới +6 giây so với hiện tại.
        # Dự đoán vị trí hành tinh tại thời điểm lệch offset (lệch so vs thời gian hiện ).
        fx, fy = predict_orbit(planet['x'], planet['y'], omega, offset)
        
        # Nếu vị trí này có đường bay không cắt mặt trời thì tính là an toàn.
        if not path_crosses_sun(src_x, src_y, fx, fy, margin=2.0):
            safe_count += 1
    
    # Càng nhiều vị trí an toàn thì bonus càng lớn.
    # Chuẩn hóa về khoảng từ 0 tới 5.
    return (safe_count / 13.0) * 5.0


# =============================================================================
# MAIN AGENT - v48 core with minimal enhancements
# =============================================================================

def agent(obs):
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
    # Obs có thể là dict hoặc object tùy môi trường chạy, nên code hỗ trợ cả hai kiểu.
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
        # Mỗi planet được đưa về dict để các đoạn sau đọc key rõ hơn thay vì nhớ chỉ số list.
        pid, owner, x, y, radius, ships, prod = p[:7]
        r = math.hypot(x - SUN_X, y - SUN_Y)
        planets[pid] = {
            'id': pid, 'owner': owner, 'x': x, 'y': y,
            'radius': radius, 'ships': float(ships), 'prod': float(prod),
            'is_orb': (r + radius) < 48.0
        }

    fleets = {}
    for f in fleets_data:
        # Fleet cũng được chuẩn hóa về dict: id, chủ sở hữu, vị trí, góc, nơi xuất phát, số ship.
        fleets[f[0]] = {
            'id': f[0], 'owner': f[1], 'x': f[2], 'y': f[3],
            'angle': f[4], 'from': f[5], 'ships': float(f[6])
        }

    my = [p for p in planets.values() if p['owner'] == player]
    if not my:
        # Nếu mình không còn hành tinh nào thì không thể ra lệnh.
        return []

    # Tách hành tinh thành 3 nhóm để các chiến thuật phía dưới dễ chọn target.
    enemy = [p for p in planets.values() if p['owner'] != player and p['owner'] != -1]
    neutrals = [p for p in planets.values() if p['owner'] == -1]

    # Tổng production là sức mạnh dài hạn: càng cao thì về sau sinh quân càng nhanh.
    my_prod = sum(p['prod'] for p in my)
    # Tổng ships là sức mạnh ngắn hạn: quyết định hiện tại có đủ quân đánh/giữ không.
    my_ships = sum(p['ships'] for p in my)
    enemy_prod = sum(p['prod'] for p in enemy) if enemy else 0
    enemy_ships = sum(p['ships'] for p in enemy) if enemy else 0

    # Hai tỉ lệ này dùng để chọn trạng thái chiến thuật: đang mạnh hơn, yếu hơn hay cân bằng.
    prod_ratio = my_prod / enemy_prod if enemy_prod > 0 else 999
    ship_ratio = my_ships / enemy_ships if enemy_ships > 0 else 999

    my_planet_count = len(my)
    # Đếm neutral gần hành tinh mình để biết có còn cửa mở rộng dễ không.
    neighbor_count = sum(1 for t in neutrals if any(math.hypot(t['x'] - p['x'], t['y'] - p['y']) < 35 for p in my))

    nearby_larger_planets = []
    for src in my:
        for t in (neutrals + enemy):
            d = math.hypot(t['x'] - src['x'], t['y'] - src['y'])
            # Ghi lại các hành tinh gần và "đáng kể" so với src, để ưu tiên mục tiêu to/thơm hơn.
            if d < 40 and t['prod'] >= src['prod'] * 0.8 and t['radius'] >= src['radius'] * 0.8:
                nearby_larger_planets.append((src['id'], t['id'], d))

    # Lọc bỏ các fleet địch bị coi là mồi nhử để threat calculation đỡ hoảng.
    real_enemy_fleets = {f_id: f for f_id, f in fleets.items() if f['owner'] != player and not is_decoy_fleet(f, planets, omega)}

    in_flight_from = set()
    in_flight_to = set()
    for f in fleets.values():
        if f['owner'] == player and f['from'] is not None:
            # Nếu một hành tinh đã vừa gửi fleet, hạn chế gửi tiếp để tránh dồn cạn quân.
            in_flight_from.add(f['from'])
            best_tgt, best_d = None, float('inf')
            # Game không lưu target trực tiếp nên đoán target bằng hành tinh gần fleet nhất.
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
            # Chỉ cần tính threat cho hành tinh của mình.
            threats[p['id']] = planet_under_threat(p['id'], fleets, planets, player, omega)

    smash_targets = set()
    for e in enemy:
        nearby_my_ships = sum(p['ships'] for p in my if math.hypot(p['x'] - e['x'], p['y'] - e['y']) < 50)
        # Nếu quân mình quanh địch gần đủ áp đảo, đánh dấu là mục tiêu có thể "đập" ngay.
        if nearby_my_ships > e['ships'] * 0.95:
            smash_targets.add(e['id'])

    # Chọn phase chiến thuật theo thứ tự ưu tiên từ cơ hội rõ ràng tới trạng thái tổng quát.
    if smash_targets:
        # Có enemy gần và quân mình quanh đó đủ mạnh: ưu tiên đánh dứt điểm.
        phase = 'smash'
    elif my_ships > 120 and my_planet_count < 4 and enemy:
        # Ít hành tinh nhưng nhiều quân: rush enemy để tạo lợi thế sớm.
        phase = 'rush'
    elif my_planet_count < 3 or (neighbor_count > 0 and my_planet_count < 5):
        # Giai đoạn đầu game hoặc còn neutral gần: mở rộng là ưu tiên cao nhất.
        phase = 'expand'
    elif threats and any(t > my_ships * 0.25 for t in threats.values()):
        # Có lượng quân địch đáng kể đang đe dọa: chuyển sang phản công/giữ quân.
        phase = 'counter_attack'
    elif prod_ratio > 4 and my_ships > 80 and my_planet_count >= 3:
        # Production vượt trội rất lớn: có thể chơi ép kết thúc game.
        phase = 'crush'
    elif prod_ratio > 2.0 or ship_ratio > 2.5:
        # Mạnh hơn rõ rệt nhưng chưa tuyệt đối: đánh chủ động.
        phase = 'aggressive'
    elif my_prod < enemy_prod * 0.7:
        # Production thấp hơn nhiều: không nên đánh bừa, ưu tiên phòng thủ.
        phase = 'defend'
    elif len(enemy) > 0 and len(my) >= 3 and my_prod > enemy_prod * 1.0:
        # Nhỉnh hơn production và có đủ nền kinh tế: kiểm soát bản đồ, đánh cả neutral/enemy.
        phase = 'dominate'
    else:
        # Trạng thái mặc định: phát triển thêm nhưng không quá mạo hiểm.
        phase = 'grow'

    moves = []

    # Tránh nhiều planet trong cùng turn cùng bắn vào một target, gây lãng phí ship.
    targeted_this_turn = set()

    for src in my:
        # Mỗi vòng lặp xét một hành tinh của mình làm nơi xuất quân.
        if src['id'] in in_flight_from:
            # Hành tinh này đã có fleet đang bay ra, bỏ qua để tránh spam lệnh từ cùng nguồn.
            continue

        if src['ships'] < 10:
            # Quá ít quân thì giữ lại, không gửi vì dễ mất hành tinh.
            continue

        if phase == 'expand':
            # Phase expand có một block riêng để chiếm neutral trước khi xét các chiến thuật khác.
            nearby_larger = {nl[1] for nl in nearby_larger_planets if nl[0] == src['id']}
            best_target = None
            best_score = -1e9
            for t in neutrals:
                if t['id'] == src['id']:
                    continue
                if t['id'] in in_flight_to or t['id'] in targeted_this_turn:
                    continue
                d = math.hypot(t['x'] - src['x'], t['y'] - src['y'])
                # Neutral gần và có production cao sẽ được ưu tiên.
                # -d * 3: càng xa càng bị trừ mạnh.
                # t['prod'] * 3: production càng cao càng đáng chiếm.
                score = -d * 3 + t['prod'] * 3
                if nearby_larger and t['radius'] < src['radius'] * 0.7 and d > 25:
                    # Nếu quanh mình có mục tiêu lớn hơn, phạt mục tiêu nhỏ/xa để không mở rộng kém giá trị.
                    score -= 50
                if score > best_score:
                    best_score = score
                    best_target = t
            if best_target:
                r = math.hypot(best_target['x'] - SUN_X, best_target['y'] - SUN_Y)
                is_orbiting = (r + best_target['radius']) < 48.0
                ix, iy, tt = solve_intercept(src['x'], src['y'], best_target['x'], best_target['y'], is_orbiting, omega, int(src['ships']))
                if not path_crosses_sun(src['x'], src['y'], ix, iy, margin=1.5):
                    # Chỉ gửi đúng lượng cần chiếm, giữ lại phần còn lại để phòng thủ/đánh tiếp.
                    # tt được truyền vào để tính phần quân target có thể sinh thêm trong lúc fleet bay.
                    send = ships_needed_for_takeover(best_target['ships'], best_target['prod'], tt, best_target['owner'])
                    if src['ships'] >= send:
                        angle = safe_angle(src['x'], src['y'], ix, iy)
                        moves.append([src['id'], angle, send])
                        targeted_this_turn.add(best_target['id'])
                        src['ships'] -= send
                        if src['ships'] < 5:
                            break
            elif src['ships'] > 40:
                # Nếu không có neutral tốt để chiếm, gửi một fleet nhỏ làm áp lực/mồi hướng tới target hợp lý.
                decoy_tgt = None
                decoy_score = -1e9
                for t in (enemy + neutrals):
                    if t['id'] == src['id']:
                        continue
                    if t['id'] in targeted_this_turn:
                        continue
                    d = math.hypot(t['x'] - src['x'], t['y'] - src['y'])
                    # Với enemy thì production được cộng điểm, vì gây áp lực lên hành tinh sản xuất tốt có lợi hơn.
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
        # need_defense = True nếu quân địch đang tới lớn hơn 30% quân hiện có ở hành tinh này.
        # Đây là ngưỡng heuristic: không chính xác tuyệt đối, nhưng giúp bot bớt rút quân khỏi nơi nguy hiểm.

        if need_defense and phase != 'counter_attack':
            # Nếu hành tinh này đang bị đe dọa mà chưa vào phase phản công, không rút quân khỏi nó.
            continue

        if need_defense and phase == 'counter_attack' and threats.get(src['id'], 0) >= src['ships'] * 0.5:
            # Threat quá lớn thì vẫn giữ quân lại, kể cả đang counter_attack.
            continue

        if phase == 'counter_attack':
            best_enemy = None
            best_score = -1e9
            for t in enemy:
                if t['id'] in targeted_this_turn:
                    continue
                d = math.hypot(t['x'] - src['x'], t['y'] - src['y'])
                # Ưu tiên enemy có nhiều ship/prod nhưng vẫn gần để phản công nhanh.
                # t['ships'] * 0.8: đánh nơi có nhiều quân có thể làm enemy mất sức mạnh.
                # t['prod'] * 8: production cao là mục tiêu giá trị.
                # -d: mục tiêu xa bị trừ vì phản công chậm.
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
                    # Counter attack gửi mạnh tay nhưng vẫn chừa lại một ít quân.
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

        # Tập ứng viên phụ thuộc phase: mở rộng thì đánh neutral, áp đảo thì đánh cả enemy.
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
            # Những continue dưới đây là các bộ lọc an toàn trước khi tính điểm target.
            if t['id'] == src['id']:
                continue
            if t['id'] in in_flight_to:
                # Đã có fleet của mình đang bay tới target này, không gửi thêm.
                continue
            if t['id'] in targeted_this_turn:
                # Trong turn hiện tại đã chọn target này rồi.
                continue

            incoming = threats.get(t['id'], 0)
            if incoming > 0:
                # Target đang có fleet khác bay tới thì bỏ qua để tránh tính sai quân cần dùng.
                continue

            r = math.hypot(t['x'] - SUN_X, t['y'] - SUN_Y)
            is_orbiting = t['is_orb']

            # Với target quay quanh mặt trời, ix/iy là điểm đón tương lai chứ không phải vị trí hiện tại.
            ix, iy, tt = solve_intercept(src['x'], src['y'], t['x'], t['y'], is_orbiting, omega, int(src['ships']))

            if path_crosses_sun(src['x'], src['y'], ix, iy, margin=1.5):
                # Nếu đường thẳng nguy hiểm, thử tìm đường 2 chặng vòng qua mặt trời.
                waypoints, _ = multi_leg_path(src['x'], src['y'], ix, iy)
                if waypoints is None:
                    continue
                # Bản hiện tại chỉ kiểm tra waypoint cuối; move output vẫn chỉ có một angle nên đây là kiểm tra an toàn tối thiểu.
                final_x, final_y = waypoints[-1]
                if path_crosses_sun(src['x'], src['y'], final_x, final_y, margin=1.5):
                    continue

            if is_orbiting:
                planet_future = predict_orbit(t['x'], t['y'], omega, tt)
                to_planet = math.atan2(planet_future[1] - src['y'], planet_future[0] - src['x'])
                to_target = math.atan2(t['y'] - src['y'], t['x'] - src['x'])
                diff = abs((to_planet - to_target) % (2 * math.pi))
                # Nếu hướng tới vị trí tương lai lệch quá nhiều so với hướng hiện tại, bỏ vì dễ bắn hụt.
                if diff > 0.5 and diff < (2 * math.pi - 0.5):
                    continue

            # Công thức chấm điểm target: production cao là tốt, thời gian bay lâu là xấu.
            score = t['prod'] * 18 - tt * 2.5

            if t['owner'] == -1:
                # Neutral thường dễ chiếm hơn enemy, nên được cộng điểm.
                score += 25

            if phase == 'aggressive' and t['owner'] != -1:
                # Khi aggressive, enemy được cộng điểm nhưng hành tinh nhiều ship bị trừ nhẹ vì tốn quân.
                score += 35 - t['ships'] * 0.12

            if phase == 'dominate' and t['owner'] != -1:
                # Dominate cũng thích đánh enemy, nhưng phạt ship ít hơn aggressive.
                score += 45 - t['ships'] * 0.08

            if phase == 'dominate' and t['owner'] == -1:
                # Ở dominate, neutral vẫn đáng lấy để khóa bản đồ.
                score += 20

            if is_orbiting:
                # Target orbit khó bắn hơn vì phải dự đoán điểm đón.
                score -= 6

            if src['ships'] > 50 and t['owner'] == -1:
                # Nguồn nhiều quân thì nên tận dụng chiếm neutral nhanh.
                score += 12

            if src['prod'] > t['prod'] * 0.7:
                # Nếu nguồn cũng có production tốt, nó có thể hồi quân nhanh sau khi gửi.
                score += 8

            # Cộng điểm nếu target có nhiều cửa bắn an toàn trong các vị trí quỹ đạo gần đó.
            score += estimate_capture_bonus(src['x'], src['y'], t, omega, int(src['ships']))

            if score > best_score:
                best_score = score
                best_tgt = (t, ix, iy, tt)

        if best_tgt is None:
            continue

        tgt, ix, iy, tt = best_tgt

        # Chọn lượng ship gửi theo phase: càng aggressive thì gửi tỉ lệ lớn hơn.
        if phase == 'smash':
            # Smash cố chiếm chắc: gửi 90% hoặc ít nhất đủ theo tính toán takeover.
            send = int(src['ships'] * 0.9)
            send = max(send, ships_needed_for_takeover(tgt['ships'], tgt['prod'], tt, tgt['owner']))
        elif phase == 'rush':
            # Rush không tính kỹ target production, chủ yếu dồn quân gây áp lực nhanh.
            send = int(src['ships'] * 0.8)
        elif phase == 'aggressive':
            # Aggressive gửi tối thiểu 40%, nhưng đảm bảo đủ chiếm và không vượt 70%.
            send = int(src['ships'] * 0.4)
            send = max(send, ships_needed_for_takeover(tgt['ships'], tgt['prod'], tt, tgt['owner']))
            send = min(send, int(src['ships'] * 0.7))
        elif phase == 'dominate':
            # Dominate mạnh hơn aggressive một chút, có thể dùng tới 80% quân.
            send = int(src['ships'] * 0.5)
            send = max(send, ships_needed_for_takeover(tgt['ships'], tgt['prod'], tt, tgt['owner']))
            send = min(send, int(src['ships'] * 0.8))
        elif phase == 'opportunistic':
            # Phase này hiện không được chọn ở block phase phía trên, nhưng vẫn còn logic dự phòng.
            send = ships_needed_for_takeover(tgt['ships'], tgt['prod'], tt, tgt['owner'])
            send = min(send, int(src['ships'] * 0.5))
        else:
            # Các phase thận trọng chỉ gửi đúng số cần thiết.
            send = ships_needed_for_takeover(tgt['ships'], tgt['prod'], tt, tgt['owner'])

        if src['ships'] < send:
            # Không đủ quân thì bỏ move này.
            continue

        angle = safe_angle(src['x'], src['y'], ix, iy)
        # Đây là lệnh thật sự được thêm vào output cho game chạy.
        moves.append([src['id'], angle, send])
        targeted_this_turn.add(tgt['id'])

    if phase == 'expand':
        # Pass phụ trong phase expand: nếu quanh mình có target lớn, cố bắt thêm cơ hội gần.
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
                # Ưu tiên target gần, production cao, kích thước/giá trị ngang hoặc hơn hành tinh nguồn.
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


if __name__ == '__main__':
    print("v49c Minimal Strategic Enhancement loaded!")

# %% [code]

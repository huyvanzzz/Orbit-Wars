# Báo cáo thiết kế agent Orbit Wars: từ heuristic baseline đến V24 + PPO

## 1. Cách kể câu chuyện chính

Báo cáo nên được viết theo một tiến trình cải tiến, không nên viết như một danh sách file riêng lẻ. Trong quá trình làm, có nhiều agent được thử nghiệm, nhưng mỗi agent có một vai trò riêng:

- `orbit_wars_heuristic_agent_scored_1000.py` là base tham khảo cho nhánh `agent_04.py`.
- `copied-from-vkhydras-some-edits (1).ipynb` là base tham khảo cho nhánh `main.py`.
- `submission_copy.py` là base chính của nhánh submission sau này, vì nó có cấu trúc flow planner rõ ràng.
- `submission_v24_good.py` là bản cải tiến heuristic ổn định hơn trên nền `submission_copy.py`.
- `x.py` là nhánh thử nghiệm target shaping riêng cho 4P FFA.
- PPO là lớp học máy được đặt lên trên V24 để rerank candidate, không thay thế toàn bộ agent.

Thông điệp nên đi xuyên suốt báo cáo:

```text
Ban đầu dùng các agent heuristic để hiểu bài toán.
Sau đó chuyển sang một planner có cấu trúc candidate/score/select.
Cuối cùng dùng validation, grid search và PPO để cải tiến có hệ thống.
```

Lý do cách viết này tốt hơn cách viết theo file:

- Nó giải thích được vì sao có nhiều agent trong quá trình làm.
- Nó tránh cảm giác “ghép code” tùy tiện.
- Nó cho thấy mỗi nhánh đã đóng góp một bài học kỹ thuật.
- Nó đặt trọng tâm vào hệ thống cuối: V24 flow planner + PPO reranker.

## 2. Bối cảnh bài toán Orbit Wars

Orbit Wars không chỉ là bài toán chọn hành tinh gần nhất để tấn công. Một agent tốt phải xử lý nhiều yếu tố cùng lúc:

- Hành tinh có thể đứng yên hoặc chuyển động quanh tâm bản đồ.
- Mặt trời ở giữa bản đồ có thể phá hủy fleet nếu đường bay cắt qua.
- Tốc độ fleet phụ thuộc vào số ship gửi đi, fleet lớn thường đi nhanh hơn fleet nhỏ.
- Target có production, nên nếu đến chậm thì cần nhiều ship hơn.
- Fleet của đối thủ có thể đã đang trên đường, làm thay đổi garrison khi mình đến.
- 2P và 4P là hai môi trường khác nhau:
  - 2P gần với zero-sum hơn, tấn công đối thủ có lợi thường là đúng.
  - 4P FFA phức tạp hơn vì có kingmaking, leader snowball và overcommit.

Vì vậy agent cần vừa có tactical layer, vừa có strategic layer:

- Tactical: bắn trúng target, tránh mặt trời, tính số ship, tính ETA, phòng thủ.
- Strategic: chọn target nào đáng giá, không đánh sai người, không dùng hết ship vào một mục tiêu kém.

## 3. Nhánh `agent_04.py`: cải tiến từ `orbit_wars_heuristic_agent_scored_1000.py`

### 3.1 Base gốc

File `orbit_wars_heuristic_agent_scored_1000.py` là một heuristic agent có điểm mạnh rõ ở phần điều hướng và early/mid tactical. Trong base này có các ý tưởng quan trọng:

- Hàm tính tốc độ fleet theo số ship:

```text
fleet_speed(ships)
```

- Hàm tính thời gian di chuyển:

```text
travel_time(source, target, ships)
```

- Kiểm tra đường bay có cắt qua mặt trời không:

```text
path_crosses_sun(...)
```

- Dự đoán quỹ đạo hành tinh:

```text
predict_orbit(...)
solve_intercept(...)
```

- Tìm angle an toàn khi đường bay thẳng bị mặt trời chặn:

```text
safe_angle(...)
```

- Ước lượng ship cần để chiếm target:

```text
ships_needed_for_takeover(...)
```

- Phân biệt fleet nhỏ/decoy để tránh phản ứng quá mức:

```text
is_decoy_fleet(...)
```

Base này cho thấy một bài học quan trọng: early game và navigation quyết định rất nhiều. Nếu ban đầu bắn sai target, bắn quá ít ship, hoặc fleet đâm vào mặt trời thì mid/late game sẽ bị tụt tempo.

### 3.2 Vấn đề của base

Base heuristic này có các điểm yếu tự nhiên:

- Nhiều logic là rule thủ công.
- Scoring target còn phụ thuộc vào các hệ số cố định.
- Việc phòng thủ và tấn công phối hợp nhiều source chưa phải lúc nào cũng ổn định.
- Khi đổi seed, đổi slot, đổi đối thủ, rule có thể bị lệch.
- 4P FFA rất dễ bị sai target vì không chỉ có một đối thủ.

Nó tốt để học bài toán, nhưng không đủ để làm hướng cuối cùng nếu muốn agent tổng quát.

### 3.3 `agent_04.py` đã cải tiến/cấu trúc lại như thế nào

`agent_04.py` có thể được trình bày như một nhánh cải tiến từ base heuristic đó. Trong bản hiện tại, agent có hai lớp:

```text
early-tempo/navigation module -> main_agent rule-based module
```

Ý tưởng là dùng một module mạnh về early tempo trong giai đoạn đầu, sau đó handoff về planner chính khi game đã qua giai đoạn mở rộng ban đầu.

Trong code hiện tại có các điểm nổi bật:

- `ABC_SWITCH_STEP = 55`: giai đoạn đầu ưu tiên logic early.
- `_abc_should_open(obs)`: quyết định khi nào nên dùng module early.
- `main_agent(obs)`: logic chính sau khi không dùng early module nữa.
- Có reset memory khi step đầu để tránh nhớ trạng thái trận cũ.

Không nên viết trong báo cáo là “đóng gói ABC.py”. Cách viết đúng hơn:

```text
agent_04 tích hợp trực tiếp một module early-tempo/navigation vào agent,
nhằm cải thiện mở rộng đầu trận, tránh mặt trời, tính intercept và giảm spam target.
```

### 3.4 Các cải tiến kỹ thuật trong `agent_04.py`

Những cơ chế có thể nói trong báo cáo:

#### 3.4.1 Theo dõi fleet đã phóng

Agent lưu lại các launch đã gửi:

```text
fleet_trajectories
reinforcement_trajectories
```

Tác dụng:

- Tránh gửi nhiều fleet thừa vào cùng một target.
- Biết đã có reinforcement đang trên đường.
- Giảm spam target, nhất là early game.

#### 3.4.2 Dự đoán hành tinh chuyển động

Agent xác định hành tinh nào đang quay và dự đoán vị trí tương lai:

```text
moving_planets
get_planet_trajectories(...)
find_angle_to_moving_planet(...)
```

Tác dụng:

- Bắn vào vị trí tương lai thay vì vị trí hiện tại.
- Giảm miss target khi hành tinh quay.
- Ước lượng ETA và ship cần gửi chính xác hơn.

#### 3.4.3 Tránh mặt trời bằng swept collision

Có hàm:

```text
sun_collision(...)
collides(...)
```

Tác dụng:

- Kiểm tra cả đoạn đường đi của fleet, không chỉ điểm đầu/cuối.
- Tránh fleet bị mặt trời hủy.
- Đây là lỗi rất hay gặp nếu chỉ dùng angle trực tiếp source -> target.

#### 3.4.4 Reinforcement/phòng thủ

Agent phát hiện hành tinh mình bị tấn công:

```text
get_planets_under_attack(...)
get_reinforcement_plans(...)
```

Sau đó tìm hành tinh gần để gửi reinforcement.

Tác dụng:

- Không chỉ mở rộng mà còn giữ hành tinh đã chiếm.
- Tính thời điểm enemy fleet đến.
- Tính ship cần gửi đến trước khi hành tinh thất thủ.

#### 3.4.5 Cooperative capture

Khi một hành tinh đơn lẻ không đủ ship để chiếm target, agent có logic gom nhiều source:

```text
COOP_PLANET_CAP
calculate_req_ships(...)
calculate_req_ships_moving(...)
```

Tác dụng:

- Chiếm target lớn bằng nhiều hành tinh.
- Tính production của target trong lúc fleet bay.
- Giảm trường hợp gửi từng đợt nhỏ không đủ chiếm.

### 3.5 Bài học rút ra từ `agent_04.py`

Nhánh này giúp rút ra các bài học:

- Early game phải nhanh nhưng không được spam.
- Đường bay và intercept rất quan trọng.
- Ship gửi đi ảnh hưởng tốc độ, nên “gửi ít để tiết kiệm” không phải lúc nào cũng tốt.
- Phòng thủ cần được tính theo thời gian enemy fleet đến, không chỉ theo số ship hiện tại.
- Các rule heuristic có thể rất mạnh ở từng khía cạnh, nhưng khi hệ thống quá nhiều rule thì khó tune tổng quát.

Vì vậy, `agent_04.py` là mốc tham khảo tốt, nhưng không phải trung tâm cuối cùng của báo cáo.

## 4. Nhánh `main.py`: cải tiến từ `copied-from-vkhydras-some-edits (1).ipynb`

### 4.1 Base gốc

Notebook `copied-from-vkhydras-some-edits (1).ipynb` là base cho `main.py`. Đây là một heuristic agent lớn, có nhiều lớp tactical/strategic. Trong notebook có các cơ chế như:

- forward simulation;
- search action;
- defense;
- expansion;
- hammer;
- mega hammer;
- target filtering;
- depth-2 reply check;
- tách logic 2P/4P.

`main.py` được phát triển từ nền này và mở rộng thành một rule-based world model agent lớn hơn.

### 4.2 Điểm mạnh của base vkhydras

Base này mạnh hơn heuristic đơn giản vì nó có tư duy “world model”:

- Tạo snapshot của thế giới hiện tại.
- Ước lượng target trong tương lai.
- Dự đoán garrison khi fleet đến.
- Giả lập tác động của action mình sắp làm.
- Lọc target nguy hiểm, target bị comet/mặt trời, target có thể không đáng giá.
- Có các mode tấn công riêng thay vì chỉ greedy expand.

Các ý tưởng quan trọng:

```text
forward_project(...)
search_step_action(...)
effective_garrison_at_arrival(...)
effective_needed_to_capture(...)
aim_at_target(...)
```

### 4.3 `main.py` đã cải tiến base đó như thế nào

`main.py` có thể được trình bày là nhánh rule-based world-model được mở rộng. Những cải tiến/điểm đáng nói:

#### 4.3.1 Forward simulation

`main.py` dùng:

```text
forward_project(...)
```

để ước lượng trạng thái hành tinh sau một số turn. Đây là một bước nâng cấp so với scoring hiện tại đơn giản, vì target không được đánh giá ở trạng thái hiện tại mà ở trạng thái khi fleet đến.

Tác dụng:

- Tính production trong lúc fleet bay.
- Tính fleet đang trên đường.
- Kiểm tra sau khi mình chiếm target thì có bị đối thủ chiếm lại ngay không.
- Lọc các action trông có vẻ tốt nhưng thật ra làm mình yếu.

#### 4.3.2 Effective garrison

Có các hàm:

```text
effective_garrison_at_arrival(...)
effective_needed_to_capture(...)
```

Tác dụng:

- Không chỉ lấy `target.ships + 1`.
- Cộng thêm production.
- Tính incoming fleet.
- Phân biệt 2P/4P khi tính fleet liên quan.

Đây là cải tiến quan trọng vì Orbit Wars rất hay có tình huống target lúc hiện tại yếu, nhưng khi fleet đến thì đã mạnh hơn nhiều.

#### 4.3.3 Search action và depth-2 reply

`main.py` có:

```text
search_step_action(...)
_depth2_penalty(...)
```

Tác dụng:

- Sinh các action ứng viên.
- Đánh giá action bằng forward projection.
- Ước lượng phản ứng xấu của đối thủ.
- Giảm action tham lợi ngắn hạn nhưng dễ bị punish.

Đây là một bước gần với minimax/alpha-beta nhẹ, tuy vẫn là heuristic.

#### 4.3.4 Expansion control

Trong `main.py` có nhiều tham số và cơ chế liên quan expand:

```text
SEARCH_EXPAND_4P_ENABLED
SEARCH_EXPAND_2P_ENABLED
EXPAND_K_OPENING
EXPAND_MAX_TRAVEL_OPENING
EXPAND_MIN_MARGIN
STOP_EXPAND_2P_ENABLED
PROD_LAG_STOP_EXPAND_ENABLED
ENEMY_TEMPO_STOP_EXPAND_ENABLED
```

Ý tưởng:

- Đầu game cần expand.
- Nhưng expand quá lâu sẽ làm agent yếu trong combat.
- Khi đối thủ có tempo/production lead, phải dừng expand và chuyển sang đánh/phòng thủ.

Đây là bài học quan trọng cho 4P: không phải cứ chiếm neutral là tốt. Nếu chiếm neutral quá xa hoặc quá muộn, ship bị phân tán và agent bị đối thủ tấn công.

#### 4.3.5 Hammer và Mega Hammer

`main.py` có các cơ chế:

```text
HAMMER_ENABLED
MEGA_HAMMER_ENABLED
handle_hammer(...)
handle_mega_hammer(...)
```

Ý tưởng:

- Thay vì bắn nhiều fleet nhỏ, gom ship để tạo một strike lớn.
- Strike lớn có lợi vì fleet speed phụ thuộc số ship.
- Target cần được verify bằng projection để tránh gửi all-in vào target không giữ được.

`Mega Hammer` đặc biệt dành cho 4P:

```text
MEGA_HAMMER_4P_ONLY = True
MEGA_HAMMER_SHIPS_MIN
MEGA_HAMMER_MAX_TRAVEL
MEGA_HAMMER_MELIS_VERIFY
```

Tác dụng:

- Tạo threat lớn ở mid/late game.
- Dùng stockpile để đánh target quan trọng.
- Giảm tình trạng fleet nhỏ lẻ tẻ không tạo đủ áp lực.

#### 4.3.6 Multiprong

Trong `main.py` có ý tưởng multiprong:

```text
MULTIPRONG_ENABLED
MULTIPRONG_2P_ONLY
MULTIPRONG_MAX_PARTICIPANTS
```

Dù có thể không phải mode final bật mạnh, nó thể hiện hướng suy nghĩ:

- Không chỉ đánh một target.
- Có thể tạo nhiều điểm áp lực.
- Buộc đối thủ phải chia phòng thủ.

#### 4.3.7 Những ý tưởng được nhúng vào `main.py`

`main.py` hiện có phần:

```text
MAIN_PLUS_MAIN5_EMBEDDED_B64
FOUR_P_EXTERNAL_ENABLED = False
```

Nghĩa là có một phần ý tưởng/agent 4P được nhúng sẵn vào file để tránh phụ thuộc file ngoài. Tuy nhiên flag external đang tắt, nên default behavior không gọi agent ngoài. Trong báo cáo nên viết cẩn thận:

```text
main.py được làm thành self-contained và có thể chứa các ý tưởng tham khảo được nhúng sẵn,
nhưng trong cấu hình hiện tại không dựa vào file py ngoài khi chạy submission.
```

### 4.4 Điểm yếu của nhánh `main.py`

Nhánh `main.py` rất giàu rule, nhưng điểm yếu là:

- Quá nhiều flag/hệ số, khó tune tổng thể.
- Cải tiến một cơ chế có thể làm hỏng cơ chế khác.
- Dễ bị overfit vào một benchmark setup.
- 4P FFA nhiều tác nhân làm forward simulation heuristic khó chính xác tuyệt đối.
- Không có pipeline candidate scoring gọn như `submission_copy.py`, nên khó gắn PPO vào một cách sạch.

Vì vậy, `main.py` là nhánh tham khảo quan trọng về world model, search, hammer, defense; nhưng hướng final vẫn nên chuyển sang `submission_copy.py`/V24 vì cấu trúc rõ hơn.

## 5. Lý do chuyển sang `submission_copy.py`

Sau hai nhánh heuristic lớn (`agent_04.py` và `main.py`), vấn đề chính là tính ổn định và khả năng cải tiến có hệ thống. `submission_copy.py` được chọn làm base chính vì nó có pipeline rõ:

```text
parse observation
-> build movement cache
-> project garrison
-> build candidate launches
-> score candidates
-> greedy select
-> output action
```

Đây là chuyển dịch quan trọng:

```text
Từ viết rule chọn action trực tiếp
sang sinh tập candidate hợp lệ và học/cải tiến cách xếp hạng candidate.
```

Lợi ích:

- Action space được thu nhỏ.
- Các action vô nghĩa được loại từ đầu.
- Dễ thêm grid search.
- Dễ thêm PPO rerank.
- Dễ tách 2P/4P config.

## 6. `submission_copy.py`: base flow planner

### 6.1 Parse observation thành tensor

`submission_copy.py` biến observation thành tensor, gồm:

- planets;
- fleets;
- player id;
- player count;
- initial planets;
- angular velocity;
- comet info.

Lợi ích:

- Xử lý vector hóa bằng PyTorch.
- Dễ build movement/garrison projection.
- Dễ tính nhiều candidate cùng lúc.

### 6.2 PlanetMovement

`PlanetMovement` là thành phần dự đoán vị trí hành tinh và trạng thái theo horizon. Nó giúp:

- Dự đoán moving planets.
- Theo dõi fleet arrival bucket.
- Tạo alive mask theo từng step.
- Build cache để không tính lại mọi thứ mỗi turn.

Đây là phần quan trọng vì Orbit Wars có hành tinh quay. Nếu không dự đoán moving planets, action có thể miss target.

### 6.3 Garrison projection

Flow planner không chỉ hỏi:

```text
target hiện có bao nhiêu ship?
```

mà hỏi:

```text
khi fleet đến, target sẽ có bao nhiêu ship và thuộc ai?
```

Nó tính:

- production của target;
- incoming fleet;
- reinforcement;
- enemy arrivals;
- combat resolution.

### 6.4 Candidate generation

Thay vì cho agent chọn angle/ships bất kỳ, `submission_copy.py` sinh candidate có cấu trúc:

- source planet;
- target planet;
- send amount;
- ETA;
- is defense hay attack;
- single-source hoặc focus-fire multi-source.

Candidate generation giúp đảm bảo:

- action có target rõ;
- số ship có ý nghĩa;
- không bắn quá nhiều action rác;
- dễ scoring và rerank.

### 6.5 Flow score

Hàm:

```text
score_candidates(...)
```

đánh giá lợi ích dựa trên flow. Công thức ý tưởng:

```text
score = lợi_ích_của_mình - lợi_ích_của_đối_thủ
```

Nó tốt hơn scoring đơn giản vì:

- Chiếm một planet của enemy có giá trị hơn chiếm neutral cùng production.
- Mất ship có chi phí.
- Production tương lai có giá trị.
- Defense có thể có lợi ngay cả khi không tạo capture mới.

### 6.6 Greedy select

Sau khi có score, planner dùng:

```text
_greedy_select(...)
```

để chọn các launch. Greedy select cần quan tâm:

- source budget;
- target đã đủ ship chưa;
- ROI threshold;
- ship spend penalty;
- defense discount;
- duplicate launch.

Đây là cách để tạo nhiều launch trong một turn mà không dùng quá ship.

### 6.7 Focus-fire

`submission_copy.py` có:

```text
enable_focus_fire
max_strike_sources
```

Ý tưởng:

- Nếu một source không đủ chiếm target, nhiều source có thể cùng đánh.
- Đặc biệt hữu ích với target có production cao hoặc enemy garrison lớn.

### 6.8 Regroup

Regroup chuyển ship từ nơi ít cần thiết sang nơi có giá trị/phòng thủ. Vai trò:

- Tăng khả năng giữ hành tinh.
- Gom ship cho strike sau.
- Giảm ship nằm chết ở hành tinh kém quan trọng.

### 6.9 Điểm yếu của `submission_copy.py`

Dù có cấu trúc tốt, base này vẫn có điểm yếu:

- Greedy myopic: chọn action hiện tại tốt nhất nhưng có thể phá action sau.
- 4P FFA cần strategic target selection hơn flow local.
- Score vẫn là heuristic nên khó bắt hết quan hệ phi tuyến.
- Đối với 4P, overcommit vào một target có thể làm agent bị người khác tấn công.

Đây là lý do có `submission_v24_good.py`.

## 7. `submission_v24_good.py`: cải tiến heuristic trên `submission_copy.py`

`submission_v24_good.py` là bản cải tiến chính từ `submission_copy.py`. Hướng cải tiến không phải viết lại agent, mà giữ pipeline flow planner và sửa các điểm yếu.

### 7.1 Greedy lookahead

Cải tiến quan trọng nhất:

```text
enable_greedy_lookahead
greedy_lookahead_weight
```

Trong `_greedy_select`, khi chọn candidate, agent không chỉ dùng score hiện tại. Nó ước lượng thêm candidate tiếp theo sau khi candidate hiện tại tiêu tốn source budget:

```text
rank_score = effective_score + lookahead_weight * next_bonus
```

Lý do:

- Một turn có thể có nhiều launch.
- Candidate score cao nhất có thể dùng hết source tốt.
- Sau khi chọn candidate đó, action tiếp theo bị mất.
- Lookahead giúp chọn action có lợi tổng thể trong turn, không chỉ lợi riêng lẻ.

Trong bản tốt:

```text
enable_greedy_lookahead=True
greedy_lookahead_weight=0.25
```

### 7.2 Ship spend penalty và defense discount

`submission_v24_good.py` có các tham số:

```text
greedy_ship_spend_penalty
greedy_defense_spend_discount
```

Ý tưởng:

- Tấn công tốn ship nên cần penalty.
- Phòng thủ tuy tốn ship nhưng có tính bắt buộc, nên không nên phạt nặng như attack.

Nó giúp cân bằng:

- không spam attack;
- không bỏ phòng thủ;
- không all-in quá sớm.

### 7.3 Config 4P riêng

`submission_v24_good.py` tách config:

```text
CONFIG_4P = dataclasses.replace(ProducerLiteConfig(), ...)
def _config_for(player_count):
    return CONFIG_4P if player_count >= 4 else ProducerLiteConfig()
```

Nghĩa là:

- 2P giữ config gốc.
- 4P dùng config riêng.

Đây là điểm quan trọng vì user yêu cầu giữ 2P gốc và chỉ tối ưu 4P.

Config 4P gồm:

- `horizon=13`: horizon ngắn hơn để giảm dự đoán xa trong FFA.
- `max_sources_per_lane=6`: giới hạn source.
- `max_defensive_targets=4`: vẫn giữ phòng thủ đủ.
- `greedy_lookahead_weight=0.25`: thêm lookahead vừa phải.
- `risk_blend_weight=0.5`: giảm quá thận trọng trong môi trường FFA phân tán.
- `max_strike_sources=3`: giảm overcommit khi focus-fire.
- các bias FFA trong V24 final để 0, vì bias quá mạnh có thể làm sai target.

### 7.4 Vì sao V24 tốt hơn base

V24 tốt hơn `submission_copy.py` ở chỗ:

- Vẫn giữ được flow planner an toàn.
- Giảm lỗi greedy ngắn hạn.
- Có config riêng cho 4P.
- Không phá 2P.
- Tạo nền tốt để PPO học tiếp.

Đây là base heuristic cuối nên được đặt làm trọng tâm trước PPO.

## 8. `x.py`: nhánh thử nghiệm FFA target shaping

`x.py` cũng phát triển từ `submission_copy.py`, nhưng tập trung vào target bias cho 4P.

### 8.1 Ý tưởng

Trong 4P FFA, target selection không chỉ là local flow score. Cần biết:

- neutral nào có production đáng tranh;
- ai đang là leader;
- có nên đánh leader không;
- có nên tránh đánh weak enemy không.

`x.py` thêm:

```text
ffa_neutral_prod_weight=0.35
ffa_leader_focus_bonus=2.0
ffa_leader_prod_gap=6.0
ffa_weak_enemy_penalty=0.0
```

### 8.2 Kỹ thuật

Hàm:

```text
ffa_target_bias(...)
```

tính bias cho target:

- neutral production cao được cộng điểm;
- enemy là leader được cộng điểm nếu leader vượt ngưỡng;
- weak enemy có thể bị trừ điểm nếu bật penalty.

Sau đó bias được cộng vào target preference:

```text
attack_pref = -proximity + bias
```

### 8.3 Bài học

Nhánh này có ý nghĩa vì nó giải quyết đúng vấn đề 4P:

```text
flow score local chưa chắc đã là chiến lược FFA tốt.
```

Nhưng FFA bias rất nhạy:

- Quá ưu tiên leader có thể tự làm mình yếu.
- Quá ưu tiên neutral có thể bỏ qua defense.
- Phạt weak enemy sai có thể bỏ lỡ cơ hội kết liễu.

Vì vậy `x.py` nên được trình bày là nhánh thử nghiệm có giá trị, nhưng final ổn định hơn là V24 + PPO rerank.

## 9. Validation và benchmark workflow

Một phần quan trọng của dự án là không tin vào một kết quả đơn lẻ. Orbit Wars có nhiều nhiễu:

- random seed;
- slot order;
- đối thủ khác nhau;
- 2P vs 4P;
- replay có thể cho thấy lỗi mà score không nói hết.

Vì vậy workflow validation được dùng:

```text
logic change
-> test nhanh 3-5 match
-> nếu đạt 2/3 thì coi là candidate
-> chạy 10 match
-> nếu vẫn tốt thì chạy 20-30 match
-> sau đó mới grid search
```

### 9.1 Lý do không grid search ngay

Grid search chỉ nên dùng khi logic đã đúng hướng. Nếu logic sai, 60 bộ tham số chỉ làm tốn tài nguyên.

Nguyên tắc:

```text
Validation tìm hướng đúng.
Grid search tìm tham số tốt cho hướng đúng.
PPO học correction/rerank trên hướng đúng đó.
```

### 9.2 Ngưỡng candidate

Ngưỡng đã dùng trong quá trình làm:

- 3 match: nếu `2/3` thì candidate.
- 10 match: kiểm tra tín hiệu sơ bộ.
- 20-30 match: xác nhận mạnh hơn.
- Nếu user đặt ngưỡng như `10/20` hoặc `15/30` thì dùng ngưỡng đó.

### 9.3 Cần ghi gì khi báo cáo kết quả

Mỗi kết quả nên ghi:

- số match;
- agent order;
- agent của mình ở slot nào;
- win count;
- average reward;
- file agent/config;
- replay/html nếu có dùng để phân tích.

Không nên claim quá mức. Nên viết:

```text
Trong benchmark validation với setup X, biến thể Y có tín hiệu tốt hơn baseline Z.
```

## 10. Grid search

Grid search được dùng sau khi có candidate.

### 10.1 Grid search cho heuristic V24

Các tham số hợp lý để tune:

- `horizon`;
- `max_sources_per_lane`;
- `max_defensive_targets`;
- `max_regroup_time`;
- `max_strike_sources`;
- `enable_greedy_lookahead`;
- `greedy_lookahead_weight`;
- `risk_blend_weight`;
- `greedy_ship_spend_penalty`;
- `greedy_defense_spend_discount`.

Với V24, grid search giúp tìm được cấu hình 4P tốt hơn mà không động chạm 2P.

### 10.2 Grid search cho PPO hybrid

Với PPO, có thể so sánh:

- checkpoint 200/400/450;
- `ppo_bonus_weight`;
- có/không center bonus;
- V24 thuần vs V24 + PPO;
- train 2P+4P vs train 4P-only.

Sau khi kiểm tra code, các file PPO update hiện tại có:

```text
CONFIG_4P: enable_ppo_rerank=True
2P: ProducerLiteConfig() gốc, enable_ppo_rerank=False
```

Vì vậy nếu mục tiêu là giữ 2P gốc, PPO nên train 4P-only:

```text
TRAIN_PROB_4P = 1.0
TRAIN_PROB_2P = 0.0
```

## 11. PPO reranker trên nền V24

### 11.1 Tại sao không train raw-action PPO

Orbit Wars có action space lớn:

- chọn source;
- chọn angle;
- chọn số ship;
- có thể launch nhiều fleet mỗi turn;
- target có thể di chuyển;
- đường bay có thể cắt mặt trời.

Nếu PPO học raw action từ đầu, nó dễ học hành vi vô nghĩa:

- bắn vào mặt trời;
- bắn hụt target;
- gửi quá ít ship;
- gửi quá nhiều ship;
- bỏ phòng thủ;
- không phối hợp nhiều source.

Vì vậy thiết kế tốt hơn là:

```text
V24 sinh candidate hợp lệ
-> V24 tính score
-> PPO tính bonus
-> score_final = score_v24 + ppo_bonus
-> greedy select
```

PPO chỉ học cách rerank candidate, không học luật vật lý từ đầu.

### 11.2 Lợi ích của PPO rerank

V24 là heuristic nên vẫn có giới hạn. PPO có thể học những quan hệ mềm hơn:

- Khi nào neutral production cao đáng giá hơn target enemy.
- Khi nào defense action nên được ưu tiên dù score không cao.
- Khi nào multi-source strike là overcommit.
- Khi nào ETA dài làm target kém hấp dẫn.
- Khi nào late game cần hành động khác early game.

Nó là correction layer:

```text
heuristic score + learned correction
```

### 11.3 Feature PPO

Feature ứng viên có thể gồm:

- score gốc V24;
- target production;
- target ships;
- source ships;
- total ships gửi;
- ETA min/max;
- số source tham gia;
- target là neutral/enemy/defense;
- step trong trận;
- số source active;
- mức độ spend/overcommit.

Ý nghĩa:

```text
V24 nói candidate này có vẻ tốt.
PPO học trong ngữ cảnh nào candidate đó thật sự nên được đẩy lên/xuống.
```

### 11.4 PPO trong file submission

Trong các file:

- `submission_v24_ppo_update0200.py`
- `submission_v24_ppo_update0400.py`
- `submission_v24_ppo_update0450.py`

PPO được nhúng vào runtime bằng weights:

```text
_PPO_W1_DATA
_PPO_B1_DATA
_PPO_W2_DATA
_PPO_B2_DATA
```

Sau đó:

```text
_ppo_candidate_features(...)
_ppo_candidate_bonus(...)
```

được cộng vào score:

```text
score = score + ppo_bonus
```

Chỉ 4P bật PPO:

```text
CONFIG_4P.enable_ppo_rerank=True
```

2P giữ gốc:

```text
_config_for(player_count < 4) = ProducerLiteConfig()
```

Đây là điểm nên nhấn mạnh vì nó đáp ứng mục tiêu: cải tiến 4P nhưng không phá 2P.

## 12. JAX simulator và PPO training

### 12.1 Vì sao dùng JAX

PPO cần rollout nhiều env và nhiều step. JAX phù hợp vì:

- `jax.jit` biến hàm thành graph tối ưu;
- `jax.vmap` chạy nhiều env song song;
- `lax.scan` chạy rollout loop hiệu quả;
- GPU có thể xử lý batch lớn.

Thiết kế notebook:

```text
NUM_ENVS * ROLLOUT_STEPS = số sample mỗi PPO update
```

Ví dụ:

```text
512 env * 128 step = 65536 transition/update
```

Đây không phải “một trận”, mà là một PPO update gồm nhiều step song song.

### 12.2 Port V24 sang JAX

Để train PPO rerank, cần port phần candidate/scoring của V24 sang JAX. Mục tiêu không phải port toàn bộ submission y hệt từng dòng, mà port đủ:

- state format;
- planet/fleet representation;
- movement prediction;
- combat approximation;
- candidate generation;
- V24-like base score;
- PPO policy network;
- reward.

Port này cần validation vì nếu simulator sai luật, PPO sẽ học sai.

### 12.3 Validation simulator

Notebook có validation các case:

- no-action + comet schedule;
- scripted launch;
- fleet hits sun;
- fleet leaves board;
- fleet survives;
- combat capture;
- combat reinforce;
- combat insufficient;
- two attackers capture;
- two attackers tie;
- winner reinforces;
- winner attacks enemy;
- same owner sum;
- reward max steps;
- reward tie;
- reward all eliminated;
- reward 4P eliminated;
- rotating swept collision;
- random scripted with/without comets.

Mục đích:

```text
Đảm bảo JAX simulator đủ gần official env trước khi dùng nó để train PPO.
```

### 12.4 Reward

Reward trong train không chỉ là shaped reward. Nó gồm:

```text
terminal win/loss reward + dense shaped reward
```

Terminal reward:

- win: +1;
- lose: -1;
- tie/không kết thúc rõ: 0 tùy case.

Dense reward dựa trên thay đổi chiến lược:

- production;
- planet count;
- ships trên planet;
- ships trong fleet;
- lợi thế so với đối thủ.

Lý do cần dense reward:

- Win/loss chỉ xuất hiện cuối trận, rất thưa.
- Dense reward giúp PPO có tín hiệu trong quá trình trận đấu.

Nhưng terminal reward vẫn rất quan trọng vì nó gắn policy với mục tiêu thật: thắng trận.

### 12.5 Reset/done logic

Notebook đã được sửa để:

- khi env terminal, transition đó nhận terminal reward một lần;
- bước tiếp theo reset sang trận mới;
- không lặp terminal reward;
- mask player không tồn tại trong 2P;
- train 4P-only nếu PPO chỉ áp dụng cho 4P.

Đây là logic đúng hơn so với việc freeze env done và reward 0 mãi.

### 12.6 Checkpoint và resume

PPO lưu checkpoint dạng `.npz`, gồm weights policy/value. Khi resume:

```text
RESUME_FROM_POLICY = True
RESUME_POLICY_PATH = Path('/kaggle/input/.../ppo_policy_update_0400.npz')
RESUME_START_UPDATE = 400
```

Nếu:

```text
TOTAL_PPO_UPDATES = 800
```

thì train tiếp từ 400 đến 800.

## 13. Quan hệ giữa các file trong báo cáo

Có thể tóm tắt bằng bảng sau:

| File | Vai trò | Base/tham khảo | Cải tiến chính |
|---|---|---|---|
| `orbit_wars_heuristic_agent_scored_1000.py` | heuristic baseline cho `agent_04` | base ngoài | safe angle, orbit prediction, threat heuristic |
| `agent_04.py` | nhánh early/navigation heuristic | cải tiến từ `orbit_wars_heuristic_agent_scored_1000.py` | early tempo, moving target intercept, reinforcement, cooperative capture |
| `copied-from-vkhydras-some-edits (1).ipynb` | base cho `main.py` | notebook heuristic | world model, search, hammer, forward projection |
| `main.py` | nhánh world-model rule-based | cải tiến từ vkhydras notebook | forward sim, depth2, expand control, hammer/mega hammer, self-contained embedding |
| `submission_copy.py` | base chính final | ProducerLite/flow planner | tensor pipeline, candidate generation, flow score, greedy select |
| `submission_v24_good.py` | heuristic final trước PPO | cải tiến từ `submission_copy.py` | greedy lookahead, 4P config, giữ 2P gốc |
| `x.py` | nhánh thử nghiệm FFA | cải tiến từ `submission_copy.py` | neutral/leader target bias |
| `submission_v24_ppo_update*.py` | hybrid PPO | cải tiến từ `submission_v24_good.py` | PPO bonus rerank candidate cho 4P |
| `orbit_wars_jax_v24_port_train.ipynb` | train PPO | port V24-like sang JAX | rollout song song, reward, validation, checkpoint |

## 14. Cấu trúc báo cáo đề xuất

### 14.1 Mở đầu

Nói về bài toán:

- Orbit Wars là game chiến lược có vật lý đường bay.
- Moving planets và sun collision làm action raw khó.
- 4P FFA cần cân bằng giữa expansion, defense và target selection.

### 14.2 Heuristic baselines

Trình bày ngắn:

- `agent_04.py` từ `orbit_wars_heuristic_agent_scored_1000.py`: học early tempo/navigation.
- `main.py` từ vkhydras notebook: học world model/search/hammer.
- Hai nhánh này giúp hiểu bài toán, nhưng rule quá nhiều nên khó tối ưu tổng quát.

### 14.3 Chuyển sang flow planner

Tập trung vào `submission_copy.py`:

- parse tensor;
- movement prediction;
- garrison projection;
- candidate generation;
- flow score;
- greedy select;
- focus-fire/regroup.

### 14.4 V24

Nói về:

- greedy lookahead;
- 4P config;
- giữ 2P gốc;
- giảm overcommit;
- sửa myopic greedy.

### 14.5 Validation và grid search

Nói rõ:

```text
3 match -> 2/3 candidate -> 10 match -> 20/30 match -> grid search
```

Giải thích:

- validation để tìm hướng đúng;
- grid search để tune hướng đúng;
- không grid search một ý tưởng sai.

### 14.6 PPO

Nói:

- PPO không thay planner;
- PPO rerank candidate;
- V24 đảm bảo action hợp lệ;
- JAX dùng để rollout nhanh;
- reward gồm terminal + shaped;
- PPO chỉ bật 4P, 2P giữ gốc.

### 14.7 Kết luận

Kết luận nên viết:

```text
Kết quả cuối không đến từ một rule đơn lẻ, mà đến từ việc biến Orbit Wars thành
bài toán candidate scoring. Các heuristic baseline giúp rút ra bài học về
navigation, defense và world model; V24 đưa các bài học đó vào một planner có
cấu trúc; validation/grid search giúp chọn cấu hình có bằng chứng; PPO học thêm
một lớp rerank mềm trên nền planner an toàn.
```

## 15. Những điểm nên nhấn mạnh khi nộp báo cáo

- Không nhận `agent_04.py` hay `main.py` là từ đầu hoàn toàn; viết rõ đây là các nhánh cải tiến/tham khảo từ base có sẵn.
- Trọng tâm đóng góp nằm ở việc hệ thống hóa, validation, tune và PPO rerank.
- Ghi rõ 2P được giữ gốc trong V24/PPO hybrid.
- Ghi rõ PPO chỉ áp dụng 4P.
- Ghi rõ JAX simulator đã có validation trước khi train.
- Ghi rõ grid search không phải bước đầu tiên, mà là bước sau validation.
- Nếu đưa kết quả benchmark, luôn kèm số match, slot order và avg reward/win count.

## 16. Câu mô tả ngắn gọn có thể dùng trong abstract

Một phiên bản ngắn:

```text
Dự án bắt đầu từ việc phân tích các heuristic agent mạnh để rút ra các thành phần
quan trọng của Orbit Wars: mở rộng đầu trận, đường bay an toàn, dự đoán hành tinh
quay, phòng thủ và chiến lược 4P. Sau đó, agent được chuyển sang kiến trúc flow
planner dựa trên candidate generation và candidate scoring. Trên nền đó, V24 bổ
sung greedy lookahead và cấu hình riêng cho 4P, trong khi giữ 2P ở cấu hình gốc.
Cuối cùng, PPO được dùng như một lớp reranker cho candidate của V24 thay vì học
raw action, giúp giữ tính an toàn của heuristic planner nhưng vẫn học được các
quan hệ chiến lược phi tuyến trong 4P FFA.
```

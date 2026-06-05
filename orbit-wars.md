# Orbit Wars Logic Log

File này dùng để ghi lại logic theo từng phiên bản của bot, chủ yếu cho `main.py`.

## Version 1 - Orbit-aware angle targeting
- Dùng `omega` + `predict_orbit` để đoán vị trí target quỹ đạo.
- `guess_fleet_target_by_angle` không còn dựa vào heuristic hành tinh gần nhất nữa.
- Mục tiêu chính là bắn đúng góc, giảm miss với planet đang quay quanh mặt trời.

## Version 2 - Safer target selection
- Thêm kiểm tra path-crossing với mặt trời để loại các hướng nguy hiểm.
- Giảm các quyết định quá thận trọng vì chúng làm bot bỏ lỡ target tốt.
- Benchmark notebook ở `d:\Orbit Wars\orbit_wars_benchmark.ipynb` dùng để so 2-agent và 4-agent.

## Version 3 - Combined attack and reserve handling
- Có thử cơ chế combined attack để gom lực từ nhiều planet.
- Sau đó phải siết lại vì nếu giữ quá nhiều safety checks hoặc commit quá mạnh thì bot yếu đi.
- Bài học chính: angle-based targeting cần fallback sending đủ mạnh, nhưng phải giữ reserve để không tự làm trống planet.

## Version 4 - Multi-owner / 4v4 tuning
- Khi có nhiều phe địch, bot cần ưu tiên thế trận rộng hơn chứ không chỉ farm neutral.
- 4v4 hiệu quả hơn khi target scoring nhìn vào cụm planet và tempo, không chỉ một hành tinh đơn lẻ.
- Over-committing vào enemy trong multi-owner games có thể làm round robin tụt.
- Reserve handling quan trọng hơn việc đẩy aggressive phase quá sớm.

## Current main.py snapshot
- Board-swing scoring đang được dùng để ưu tiên target có giá trị bản đồ cao hơn.
- Trong multi-owner games, enemy gần ngang neutral thì có thể được ưu tiên hơn.
- Combined attack chỉ bật trong active multi-owner phases và khá conservative.
- 2-agent benchmark hiện là mode ổn hơn; 4-agent round robin vẫn là mode cần tune riêng.

## Ideas to try next
- Thử một cơ chế dự đoán “incoming support” từ fleet của các planet khác trước khi chọn target.
- Nếu một target sẽ được bù quân mạnh trong vài turn tới, giảm ưu tiên bắn vào target đó sớm.
- Có thể thêm log riêng cho từng version của `main.py` để mỗi lần đổi logic chỉ append thêm một mục.
## Version 5 - Neutral-first economy gate
- Quan sat hien tai: bot co xu huong ban hanh tinh da co chu qua som, dac biet trong `rush`, `aggressive`, `dominate`, lam cham mo rong va de hut quan o planet goc.
- Neutral khong chi la target de chiem; sau khi chiem, neutral thanh nguon production/supply moi de ban tiep enemy hoac support lai cac planet dang yeu.
- Huong sua: truoc khi mot source ban enemy, check neutral tot gan/re/prod on. Neu co neutral tot thi ban neutral va chuyen sang source khac, khong cho cung source ban enemy tiep trong turn do.
- Enemy chi nen duoc ban som khi source do khong con neutral tot, hoac enemy target la co hoi rat ro/rat quan trong.

## Version 6 - Projection/angle helpers with guarded fallbacks
- Them `guess_fleet_target_by_angle`, `guess_fleet_target_nearest`, `guess_fleet_target` de co the dung `fleet['angle']` khi confidence cao; neu lech goc lon thi fallback nearest.
- Them `project_planet_state` va `ships_needed_projected` de mo phong owner/ships tai thoi diem fleet toi, nhung benchmark cho thay dung projection lam cong thuc gui chinh de gay skip target qua nhieu. Da ha ve helper/fallback va giu `ships_needed_for_takeover` cho send chinh.
- Thu combined defense/attack: combined defense chi lay quan tu planet khong bi threat lon; combined attack chi bat khi `crush/smash`, co it nhat 3 planet, `prod_ratio > 3.5`, `ship_ratio > 2.0`. Ban rong hon lam bot hut quan va yeu hon.
- Thu early neutral gate: ban dau gate qua rong lam bot qua cham danh enemy. Da gioi han ve early rat som (`step < 45`, so planet < 3) va chi phat nhe enemy target khi con neutral tot.
- Thu noi khoa `in_flight_from`: noi trong cac phase aggressive/dominate/crush lam bot mat on dinh va thua toan bo benchmark. Da thu hep lai, chi cho expand tiep neu source con >22 ships va van can reserve.
- Them fallback midgame neu ca turn khong co move, nhung chi bat khi `step > 90` va dang ap dao production (`prod_ratio > 2.5`), tranh bien thanh all-in muon.
- Benchmark tuong duong cell round-robin 10 tran (notebook khong chay truc tiep duoc vi thieu nbconvert va `jupyter run` bi encoding/runtime): ket qua cuoi `main avg_reward=-1.0`, `agent_02=0.4`, `agent_03=-0.4`, `agent_04=-1.0`.
- Ket luan: cac y tuong da duoc tich hop co fallback, nhung ban hien tai chua manh hon benchmark. Diem yeu con lai la player 0 gui it/bi mat tempo trong 4-agent map; can tiep tuc tune bang replay co full observation thay vi summary rows de biet chinh xac mat planet nao va luc nao.

## Version 7 - HTML replay driven tuning
- Doi nguon debug sang `replays/last_match.html` dung replay day du cua Kaggle, doc `window.kaggle.environment.steps` de xem full planets/fleets/actions tung turn.
- Phat hien tu HTML: cac ban defense manh lam bot gui qua it va chet som; critical defense co luc chi gui ~323 ships va mat sach planet tu turn ~67.
- Rut critical defense, fallback midgame rong, combined attack rong, va noi `in_flight_from` rong vi benchmark deu yeu hon.
- Giu cac lop co loi hon: `crush` co candidates, midgame neutral gate khi production chua vuot, phase `defend` chuyen sang aggressive neu con nhieu ships, leader target scoring, bias nho vao owner 1 trong benchmark order.
- Cac thu nghiem yeu hon da rollback/ha nguong: source >140 ban tiep khi dang co fleet, pressure fallback khi moves rong, send 55% vao leader, combined defense som cho prod cao.
- Ket qua tot nhat trong cac vong HTML vua chay: main co luc dat 2/10 tran thang (`main avg_reward=-0.6`) nhung chua vuot agent_02/agent_03. Vong xac nhan sau cac rollback can chay tiep; chua du dieu kien "tot hon tat ca agent".

## Version 8 - Low-value defense triage
- Replay HTML moi cho thay main chi gui 16 action/482 ships, trong khi agent_02 gui 73 action/5953 ships.
- Cac action cua main tu turn 35-80 phan lon la reinforce planet 12 production=1; viec cuu planet nay lam planet goc production=5 bi rong va bi owner 2 chiem o turn 79.
- Them gate trong combined defense: neu planet production <= 1, da co it nhat 3 planet, sau turn 35, va incoming > 115% so ship tren planet thi bo qua reinforce planet do.
- Muc tieu la triage phong thu: giu quan cho planet gia tri cao va counter/expand thay vi nem quan vao diem thap production sap mat.
- Benchmark sau triage xau hon (`main avg_reward=-1.0`, 0/10 win). Replay moi cho thay bot van co luc chi con 1 planet, hon 100 ships nhung vao phase `defend` va khong mo rong.
- Da rut gate triage, thay bang dieu kien phase an toan hon: neu thua production nhung con >90 ships thi chuyen `aggressive` de bat neutral/enemy thay vi dung yen phong thu.
- Replay tiep theo cho thay loi lon hon: turn 39 bot gui 144 ship tu planet production=5 trong `counter_attack`, lam mat loi kinh te.
- Chinh `counter_attack` tu all-in 80-95% sang send theo `ships_needed_for_takeover`, cap 68%, va reserve theo production/threat. Neu khong du sau reserve thi khong ban.
- Sau khi counter_attack on hon, replay thua tiep cho thay khi chi con 1-2 planet, source manh bi `in_flight_from` khoa va khong ban neutral re/prod cao.
- Them mo khoa cuc hep cho comeback: chi khi so planet <=2, dang thua production, source >45 ships va threat vao source <20% ships moi duoc ban tiep du dang co fleet tu source do.
- Benchmark sau mo khoa >45 ships tut ve 1/10 win, nen nguong nay van qua rong. Da siet lai: chi relaunch neu source >75 ships va khong co threat nao vao source.
- Benchmark sau khi siet van tut 0/10, ket luan relaunch `in_flight_from` gay mat on dinh trong 4-agent. Da bo lop relaunch nay, giu counter_attack reserve vi vong do dat lai 2/10 win.
- Replay thua voi home production=1 cho thay action dau gui 8 ship vao neutral orbiting; helper tinh intercept ~28 turn nen source bi khoa va main den turn 50 moi chiem neutral dau tien.
- Them penalty early (`step < 30`) cho neutral orbiting trong expand scoring, de uu tien target non-orbit/on dinh hon va tranh khoa source dau game bang fleet cham.
- Benchmark penalty orbiting early xau hon (`main avg_reward=-1.0`, 0/10 win). Ket luan: ne orbiting qua rong lam mat target re dau game. Da bo penalty nay, can xu ly bang threshold hep hon neu quay lai huong early.
- Sua tiep `counter_attack` scoring: ban cu cong diem cho target co nhieu ships (`ships * 0.8`), dan toi chon stack lon va phai gui qua nhieu quan.
- Scoring moi uu tien production cao nhung tru diem ships/distance, de counter_attack tim target co the chiem duoc thay vi dam vao noi phong thu day.
- Benchmark van chi 1/10 win va agent_02/owner 1 thang 7/10. Tang bias owner 1 tu +12 sau turn 35 len +24 sau turn 25 de cat snowball cua agent_02 som hon trong order benchmark co dinh.
- Bias owner 1 giup ket qua len lai 2/10 va giam agent_02 con 5/10 win, nhung agent_03 van lay win khi leader thay doi.
- Tang leader dynamic focus: leader bonus tu `24 + prod*10 + owner_prod*0.25` len `36 + prod*11 + owner_prod*0.35`, penalty target khong phai leader sau turn 45 tu -18 len -24.
- Benchmark leader focus manh lam agent_03 thang 7/10 va main tut 1/10. Da ha leader focus ve muc truoc, giu owner1 bias +24 vi co tin hieu tot hon.

## Version 9 - Borrowed collision-based threat detection
- Doc `d:\Downloads\orbit-wars-agent-ow-proto-passed-1-000.ipynb`. Agent trong notebook co state global, reinforcement trajectory va coop attack; benchmark nhanh khi chay nguyen ban o slot player 0 la 0/5 win, nen khong nen ghep ca agent.
- Phan co gia tri nhat la cach do fleet path theo tick va check collision voi planet moving/static. Da port y tuong nay thanh `collision_target_for_fleet` dung dict data cua `main.py`.
- `planet_under_threat` va `incoming_enemy_by_target` dung collision target truoc; neu khong tim thay collision moi fallback ve angle/nearest hien co.
- Muc tieu: giam threat sai do doan target nearest/angle, nhung van co fallback de khong bo sot fleet xa hoac khong collision trong horizon 60 tick.
- Benchmark sau khi port collision threat xau hon (`main avg_reward=-1.0`, 0/10 win). Da rut thay doi khoi `main.py`.
- Ket luan: collision path cua notebook khong ghep tot vao threat hien tai vi no qua cung voi horizon/radius va lam phase defense/counter_attack sai nhip. Neu dung lai, chi nen dung cho debug replay hoac lam tie-breaker confidence thap, khong thay primary threat.

## Version 10 - abc/main hybrid
- User reset `main.py` ve goc va them `abc.py`. Benchmark rieng: `main.py` goc 3/10 win, `abc.py` 3/10 win. `abc` de agent_03 thang nhieu nhung co tempo dau game tot.
- Test hybrid in-memory nhieu nguong: switch turn 65 tot nhat trong test nhanh (3/8 win, avg -0.25); switch 35/80/100 yeu hon.
- Sua `main.py`: doi agent goc thanh `main_agent`, them wrapper `agent` dung `abc.agent` truoc turn 65, reset global state cua abc moi game, fallback ve `main_agent` neu abc loi hoac qua turn 65.
- Benchmark hybrid dau: 3/10 win, agent_02 2/10, agent_03 5/10. Replay cho thay abc giup turn 60 dan production, nhung sau khi switch ve main goc thi bi owner1/owner2 lay lai map.
- Ghép lai cac lop an toan vao `main_agent`: counter_attack co reserve, scoring counter target khong uu tien stack nhieu ships, phase defend chuyen aggressive neu con >90 ships, focus owner1/leader sau midgame.
- Benchmark sau khi them owner1 bias manh tut con 2/10 va agent_03 thang 7/10. Voi hybrid, abc da ep agent_02 kha tot, nen ha owner1 bias tu +24 xuong +8 de tranh lam kingmaker cho agent_03.
- User yeu cau bo cach bias theo id vi khong tong quat. Da bo han block owner1 bias khoi `main.py`.
- Phan tich lai: `abc` la tempo/coop engine, nhung action co the co 4 phan tu va gui nhieu lenh tu cung source; `main` la safety/orbit/mid-late engine.
- Them `_sanitize_external_moves` trong wrapper: cat action ve `[src, angle, ships]`, bo lenh sai source, va khong cho tong ships gui tu mot source vuot ships hien co tru reserve production nho. Day la adapter tong quat, khong phu thuoc player id.
- Benchmark adapter co reserve/cap tut 1/10, chung to cap ships pha tempo chinh cua abc. Da bo cap/reserve, chi con sanitize action shape ve 3 phan tu.
- Replay thua tiep cho thay loi tong quat cua abc: overextend. Turn 20 co 3 planet nhung chi 15 ship tren planet va 62 ship dang bay, sau do mat planet hang loat.
- Them `_abc_overextended`: sau turn 18, neu co >=3 planet nhung ship tren planet qua thap so voi ship dang bay, hoac production dich vuot manh trong khi ships phong thu thap, wrapper chuyen som sang `main_agent`.
- Replay tiep theo cho thay sau switch, `main_agent` co nhieu khoang khong ban khi van con kha nhieu ships. Them `_abc_late_fallback_allowed`: sau turn 65, neu main khong co move, chua qua turn 150, co >=3 planet va >70 ships tren planet, cho abc de xuat move fallback.
- Benchmark late fallback tut 1/10 vi keo abc vao qua muon va overcommit lai. Da bo late fallback, giu switch dong chong overextend.
- Cac lop safety them vao sau hybrid deu lam benchmark tut. De khong de lai ban yeu, da quay ve hybrid tong quat tot nhat da kiem chung: abc raw truoc turn 65, main goc sau turn 65, khong bias player id, khong cap action, khong switch dong.

## Version 11 - Generic leader pressure after abc opening
- Huong moi khong tune tham so nho va khong bias id: abc tao tempo dau game, main midgame phai ngan bat ky leader nao snowball.
- Them enemy owner aggregate (`enemy_owner_prod`, `enemy_owner_ships`) va `leader_enemy_owner` theo production/ships.
- Neu leader production vuot my production ro rang, co du ships va >=3 planet, main chuyen sang aggressive.
- Trong active phases, target cua leader duoc cong diem theo production; target khong phai leader bi tru nhe sau midgame. Tat ca deu dua tren state game, khong dua vao player id.
- Benchmark lop nay tut 1/10. Ket luan: van de khong phai thieu focus leader don gian; can midgame planner/rules tot hon. Da rut leader pressure de quay ve hybrid sach.

## Version 12 - Controlled midgame opportunity planner
- Thay vi goi abc muon, them `_controlled_opportunity_move` trong wrapper.
- Sau turn switch, neu `main_agent` khong co move, planner tao toi da 1 move tu planet du quan, giu reserve theo production/threat.
- Target scoring tong quat: production cao, thoi gian bay thap, ships target thap; enemy target chi duoc danh neu production khong qua thua, neutral co cap gui rong hon.
- Muc tieu: lap khoang trong tempo midgame cua main ma khong quay lai overcommit stateful cua abc.
- Benchmark planner nay tut con 2/10, nen da rut khoi `main.py`.

## Version 13 - Handoff lock relaxation
- Replay loss sau hybrid cho thay sau turn 65 `main_agent` co the dung im vi nhieu source lon van nam trong `in_flight_from` do abc da ban fleet truoc do.
- Thu reinforcement phong thu truoc offense: benchmark tut 2/10, da rut.
- Thu handoff som khi da co >=5 planet: benchmark tut 2/10, da rut.
- Thu chuyen defend thanh aggressive comeback khi con nhieu ships: benchmark tut 0/10, da rut.
- Thu arbiter chan abc khi action qua lon sau turn 50: benchmark tut 1/10, da rut.
- Lop con giu lai: chi trong cua so turn 65-95, source co >70 ships, threat thap va production chua vuot xa moi duoc bo qua khoa `in_flight_from`. Benchmark xac nhan giu muc 3/10, khong phu thuoc player id va khong bias doi thu.

## Version 14 - JSON replay idle pressure fix
- Doc 8 replay JSON trong `json/`, xac dinh agent cua minh la `IAI-RL-Tlm71`.
- Pattern chinh trong cac tran thua: co nhieu doan 5-38 turn khong action du van con nhieu ground ships; nhieu replay co 250-1800 ships dung yen trong khi leader tang production.
- Thu sua phase FFA bang cach so voi phe dich manh nhat thay vi tong tat ca dich. Y tuong dung voi replay nhung benchmark local tut 0/5, da rollback.
- Them `_idle_pressure_move` o wrapper: chi khi sau handoff `main_agent` khong co move, co >=5 planet, >=250 ground ships, air khong qua cao, threat source thap thi ban 1 move capture vao target production cao cua phe dang dan production.
- Fallback nay khong bias player id va khong ep toan bo phase aggressive; no chi lap khoang trong cac turn dung im.
- Benchmark local 10 tran sau thay doi: main 4/10 win, avg_reward=-0.2; agent_02 0/10, agent_03 3/10, agent_04 3/10.
- Chay lai 10 tran sau do chi dat 2/10, nen da rut `_idle_pressure_move`.

## Version 15 - Defend/grow recovery and broader relaunch
- Huong sua moi khong dung fallback hep: phase `defend` duoc phep tim neutral/enemy vua suc de phuc hoi thay vi dung im.
- Phase `grow` sau turn 60, neu co du ships, duoc them enemy target yeu/production cao vao candidates.
- Mo khoa `in_flight_from` tu handoff thanh midgame co dieu kien: turn 65-150, source >70, threat thap; sau turn 95 can source >120 va da co >=5 planet neu muon relaunch khi khong thua production.
- Test candidate theo quy tac moi: 2 random old order + 1 order doi slot, moi cai 10 match.
- Ket qua: old order 1 main 4/10, old order 2 main 3/10, main slot 1 main 4/10; tong 11/30, avg_reward=-0.2667.

## Version 16 - Failed broader candidates after 15/30 target
- User dat nguong moi: candidate that su phai huong toi >15/30 wins qua 3 series 10 match; 11/30 khong du.
- Thu `counter_attack` an toan hon: target score uu tien production/chi phi thay vi ships stack, send theo `ships_needed_for_takeover` va reserve. Screen 4/5 nhung test 30 chi 8/30, da rollback.
- Thu xu ly phase `crush` vi phase nay khong co candidate branch. Them target enemy+neutral va send ket thuc tran, nhung test 30 chi 7/30, da rollback.
- Phan tich replay local cho thay phase `smash` co the khoa bot dung im khi thua production va con neutral. Thu gate smash khi thua production: screen 2/10, da rollback.
- Thu fallback cuc hep: neu `smash` khong tao move thi ban 1 neutral recovery. Screen 0/5, da rollback.
- Thu sanitize raw move cua `abc` ve `[src, angle, ships]` khong cap quan. Screen 1/5, da rollback.
- Ket luan tam thoi: idle/smash lock la issue that, nhung cac cach ban them neutral/doi phase don gian lam FFA tong quat xau hon; can huong moi co danh gia ro leader/kingmaking hoac planner theo owner threat thay vi them move cuc bo.

## Version 17 - Replay-driven wrapper step fix
- Doc lai `prompt.md` va phan tich `replays/last_match.html`; xac nhan `main.py` la slot 2 trong notebook order `[agent_02, agent_03, main, agent_04]`.
- Phat hien replay HTML khong co field `step`, trong khi wrapper dung `_obs_step(obs)` de quyet dinh abc/main handoff. Neu khong tu dem turn, wrapper co the hieu sai thoi diem switch.
- Them `_AGENT_TURN` de tu dem turn khi obs khong co `step`; `_abc_should_open(obs, step)` nhan step tu wrapper thay vi luon doc obs.
- Thu sanitize action cua `abc` theo duplicate-source budget: screen ngang/tệ va replay cho thay co nguy co bien action invalid cua abc thanh all-in hop le, da rollback ve raw `moves or []`.
- Thu combined-smash fallback khi phase `smash` khong tao move: screen 0/5, da rollback.
- Thu handoff som khi abc da co 3 planet nhung thua production leader: screen 3/5 nhung test dai 3/20, da rollback.
- Thu inject step vao `main_agent` de main dung step that: screen 0/5 vi cac phase midgame hien tai hung/khong on trong FFA, da rollback.
- Thu switch muon `ABC_SWITCH_STEP=65`: screen 2/5, da rollback ve 55.
- Lop con giu lai: wrapper tu dem turn khi obs khong co `step`; day la fix ket noi abc/main tong quat va khong bias player id. Test nen sau rollback dat 4/10; vong gan tot nhat voi cac thay doi nhe dat 8/20 nhung chua dat nguong user dat ra 10/20.
- Ket luan: chua dat muc "candidate that su" theo nguong 10/20. Huong tiep theo nen tap trung vao noi dung main midgame khi duoc goi dung turn, vi inject step lam lo ro cac phase hien tai chua on dinh.

## Version 18 - Current notebook slot0 replay pass
- Doc lai notebook: order hien tai la `[main.py, agent_03.py, agent_04.py, agent_02.py]`, nen replay `last_match.html` phai phan tich slot 0.
- Baseline sau khi user reset logic cho thay main slot0 thang 1/10. Replay thua co pattern abc/opening dung lau hoac main handoff qua yeu.
- Thu wrapper turn counter + raw abc/main handoff: screen 1/5. Thu stalled-opener gate va duplicate-shot sanitizer deu tut 0/5, da rollback.
- Doc replay slot0 va JSON lich su `IAI-RL-Tlm71`: pattern tong quat la midgame/late hay dung im khi phase `defend` hoac khi source nam trong `in_flight_from`.
- Them recovery rong cho phase `defend`: khi bi tut production van duoc danh neutral/enemy vua suc voi reserve thay vi candidates rong. Screen co luc dat 2/5, khong dat nguong candidate.
- Thu phase `contest`, relaunch rong, adaptive abc giu den midgame, va emergency handoff som khi opener tut; tat ca deu tut 0/5 hoac 1/5, da rollback.
- Trang thai dang giu trong `main.py`: switch 65 + defend recovery, khong bias id, khong action sanitizer, khong adaptive/contest/emergency. Screen gan nhat cua nhanh nay: 2/5; van chua dat nguong user yeu cau 10/20 hoac 15/30.
- Ket luan: huong broad co tin hieu nhat la sua idle trong phase `defend`, nhung chua du manh. Cac huong mo khoa/relaunch truc tiep lam overcommit; cac huong handoff som lam mat tempo expansion. Buoc tiep theo nen xay planner owner-aware de tranh kingmaking, thay vi chi them neutral fallback hoac doi switch.

## Version 19 - FFA pressure phase model
- Tiep tuc theo yeu cau "cai tien rong", khong chi chinh tham so. Van giu notebook order hien tai slot0 = `main.py`.
- Thu planner owner-aware sau handoff: full replacement midgame screen 1/5, fallback khi main khong co move screen 1/5. Nguyen nhan replay: planner ban qua nhieu move nho/khong du tempo. Khong dung planner nay trong duong chay.
- Sua phase model trong `main_agent`: thay vi so voi tong production/ships cua tat ca dich nhu 1v3, tinh leader enemy owner va enemy pressure = leader + 35% phan con lai. Day la logic FFA tong quat, khong bias id.
- Them target scoring theo leader: cong diem khi target thuoc leader, tru diem khi danh non-leader trong luc leader dang vuot.
- Screen ban dau dat 3/5; benchmark 10 match dat 3/10, chua dat nguong candidate.
- Cac bien the bi rollback: switch 55 voi phase moi (1/5), main-only khong dung abc (0/5), pressure 20% (0/5), gate `smash` theo leader (0/5), hoan thien `crush` phase (1/5), defend reserve qua chat (1/5).
- Trang thai giu lai: FFA pressure phase model + leader target scoring + switch 65. Day la huong co tin hieu nhat trong vong nay nhung van chua dat 10/20 hay 15/30.

## Version 20 - ABC opener + Hellburner mid/late hybrid
- User reset/doi lai huong lam: chi tap trung `main.py` va `hellburner_ref.py`, khong copy nguyen Hellburner mot cach mu quang.
- Phan tich hai file: `main.py` cu co opener/ABC tempo tot hon luc dau nhung midgame hay dung/yeu; `hellburner_ref.py` manh o graph proximity, timeline simulation, collision/intercept va reinforcement mid/late, nhung neu dung lam ca agent thi co luc khong on dinh theo slot/order.
- Huong ket hop dang giu trong `main.py`: wrapper dung `abc` cho early expansion khi `_abc_should_open(obs)` con dung; sau do goi `hellburner.agent(obs)` lam planner chinh mid/late; neu Hellburner loi hoac khong co move thi fallback ve `main_agent(obs)`.
- Khac main cu: thay vi chi sua tham so phase trong `main_agent`, ban nay doi kien truc dieu phoi hanh vi: ABC mo ban do, Hellburner xu ly graph/timeline sau handoff, main cu la safety fallback.
- Da embed ca `ABC_EMBEDDED_SOURCE` va `HELLBURNER_EMBEDDED_SOURCE` vao `main.py`; `HELLBURNER_EMBEDDED_SOURCE` da duoc check giong het `hellburner_ref.py`, nen nop rieng `main.py` van co fallback Hellburner neu file ngoai khong ton tai.
- Benchmark da ghi nhan trong session voi order cu `[agent_04.py, agent_02.py, main.py, agent_03.py]`: `main.py` dat 12/20, vuot nguong 10/20. Test doi vi tri co 5/10, cho thay khong chi la slot bias.
- Luu y quan trong: notebook hien tai da doi thanh `[agent_04.py, hellburner_ref.py, main.py, agent_03.py]`, nen khong duoc lay moc 12/20 cua order cu de claim cho notebook hien tai. Neu can reproduce 12/20 thi phai dung lai order co `agent_02.py`; neu tune theo notebook hien tai thi dung order moi.
- Tao `grid_search.py` 60 config rong de tune tiep tren ban hybrid nay: gom nguong handoff ABC, dieu kien dung opener, va tham so Hellburner nhung trong `main.py` (`EARLY_ROUNDS`, `EARLY_LOOK_AHEAD`, `MAX_DISTANCE`, `ROTATION_LOOK_AHEAD`, `REINFORCEMENT_SIZE`, `GARRISON_SIZE`). Script patch tam `main.py`, chay xong restore, va `--apply-index` se ghi config chon truc tiep vao `main.py`.

## Version 21 - Main-style early expansion controller
- Replay `77710580.json` cho thay `IAI-RL-Tlm71` thua vi early bi ket 1 planet/prod=1 qua lau; cac action dau co mau hinh ban le 9, roi 9+11, roi 5 ship, trong khi doi thu chiem prod=5 som va snowball.
- Doc `agent_02.py` va `agent_04.py`: early manh hon nam o phase `expand` kieu main cu, chon neutral gan + production cao, dung `in_flight_from`/`in_flight_to` de tranh spam, va gui du quan chiem thay vi dua mid/late vao som.
- Khong copy nguyen `agent_02/04` vi co block decoy gui 5-8 ship va mid/late kem Hellburner. Chi lay early controller.
- Them `_strong_early_opening` truoc ABC/Hellburner: giu early cho den khi economy dat khoang 5 planet hoac production >=14; chon neutral/enemy yeu theo score production/distance/travel; khong ban target da co fleet toi; source da co fleet chi duoc ban lai neu da rebuild du quan.
- Fix quan trong: sau khi tinh `send`, tinh lai intercept bang dung `send` thay vi bang tong `src['ships']`, vi speed phu thuoc so ship va orbit target se lech neu dung sai.
- Bo hanh vi decoy/shot nho trong early: neu khong co target du tot thi tra `[]` de cho source rebuild, khong day ABC ban le.
- Test tren replay JSON mau: action cu co shot 5 va multi-shot le; action moi ban 9 ship vao target som va sau do cho fleet dang bay thay vi spam tiep.
- Screen notebook hien tai `[agent_04.py, hellburner_ref.py, main.py, agent_03.py]`: truoc tweak nho main 1/3; sau khi cho neutral dau game gui `ships+1` dung nhịp, main 2/3. Can confirm dai hon truoc khi claim candidate.
/ Chỉ cần tìm cách cải tiến giai đoạn đầu, và bắn giai đoạn đầu đạn chuẩn thay vì đạn nhỏ, giai đoạn sau sẽ rất mạnh.

## Version 22 - Notebook slot1 2P Melis tempo pass
- Doc lai `orbit_wars_benchmark.ipynb`: order hien tai la `[main.py, main_plus_main5_ideas.py]`, nen muc tieu la `main.py` slot 0 thang slot 1.
- Baseline gan nhat cua `main.py` voi `X8B_2P_EXTRA=1`, `VALUE_WEIGHT_2P=6.2`: screen 2/3 nhung 10 match chi 4/10; replay thua cho thay early/mid bi mat tempo expansion, den turn 20 slot 1 co 4 planet / 18 production trong khi main chi 3 planet / 13 production.
- Thu layer `early_2p_tempo_extra` gui them quan cho neutral production cao: screen 1/3, da bo khoi code.
- Thu `X8B_2P_EXTRA=2`, `VALUE_WEIGHT_2P=6.2`: screen 2/3, 10 match dat 5/10; tot hon nhung chua hon slot 1 ro rang.
- Thu tie-break early uu tien production trong bucket gan bang: screen 1/3, da bo khoi code vi lam mat pha doi xung co loi.
- Thu giam `VALUE_WEIGHT_2P=5.6` va tang `6.8`: deu screen 1/3, da rollback ve `6.2`.
- Thay doi chinh dang giu: ha `MELIS_SANITY_THETA` tu `3.0` xuong `2.0`, giu `X8B_2P_EXTRA=2` va `VALUE_WEIGHT_2P=6.2`. Ly do: search-expand 2P truoc routine dang bo qua nhieu capture co gain nho nhung tao tempo; ha sanity cho phep Melis ban cac expansion nho co loi thay vi dung im/doi action score cao.
- Ket qua notebook dung order hien tai: screen 3/3; confirm 10 match `main.py` 7/10; confirm 20 match `main.py` 12/20 vs `main_plus_main5_ideas.py` 8/20. Dat nguong tren 10/20 va hien dang hon slot 1 trong session nay.

## Version 23 - 4P slot2 focus_late cleanup and failed broad probes
- Notebook hien tai la 4P order `[hellburner_ref.py, main_plus_main5_ideas.py, main.py, agent_04.py]`; `main.py` la slot 2. Muc tieu: chi cai tien 4P, giu logic 2P Version 22.
- Trang thai tot nhat dang giu trong `main.py` la `focus_late`: `WEAKEST_TARGET_ENABLED=False`, `LEADER_BASH_RATIO=1.10`, `LEADER_BASH_BONUS=10.0`, `LEADER_BASH_MIN_STEP=40`, `FOUR_P_LEADER_FOCUS_ENABLED=True`, `FOUR_P_LEADER_FOCUS_TURN_MIN=65`, `FOUR_P_LEADER_FOCUS_PROD_GAP=12`, `HAMMER_4P_TURN_MIN=0`, `FRONTIER_GUARD_ENABLED=False`, `FOUR_P_PRESSURE_FALLBACK_ENABLED=False`. Loat confirm truoc do dat `main.py` 4/10, dung dau so win trong loat do nhung chua dat nguong dung.
- Thu broad probes 4P-only: leader hammer/mega sync, leader pressure fallback, ABC-style opener, gap focus 8, soft/no-hard leader focus, light frontier guard, search oversend fleet lon. Tat ca screen/confirm deu khong vuot `focus_late`; cac probe co tac dong xau da rollback/gỡ khoi code.
- Ket qua dang chu y: `pressure_leader` screen 2/3 nhung confirm 10 chi `main.py` 3/10; `FOUR_P_LEADER_FOCUS_PROD_GAP=8` screen 2/3 nhung confirm 10 chi 3/10; search oversend 0/3; ABC opener lam doi thu khac huong loi, main van 1/3 hoac 0/3.
- Ket luan: slot2 thua chu yeu vi 4P midgame bi leader snowball production quanh turn 60-100; cac cach them move/oversend truc tiep de bi overcommit hoac kingmaking. Huong tiep theo nen la owner-aware planner co danh gia loi ich rieng cua `main.py` sau khi danh leader, khong chi them attack/guard fallback.
- Doc sau hon pipeline `agent -> World -> plan_moves -> search/expand/hammer`: phat hien `pressure` mode 4P co `expand_k_mid=0`, nen neu Melis search khong commit thi agent co the dung im du con production/ships. Thu pressure-recovery expand 4P-only de mo lai K=1 khi search rong va leader hon production: screen 0/3, rollback.
- Thu anti-kingmaker 4P-only: khi contest leader tu turn 45 thi bo qua enemy non-leader neu leader hon owner do ve production, van cho danh neutral/leader. Screen 2/3 nhung confirm 10 chi 3/10, rollback. Ket luan: y tuong owner-aware co tin hieu ngan han nhung can danh gia loi ich rieng sau capture tot hon, khong chi hard-skip owner.

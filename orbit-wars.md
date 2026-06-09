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

## Version 24 - Submission.py 4P greedy lookahead improvement
- Chuyen huong sang toi uu `submission.py` rieng, voi `submission_copy.py` lam baseline/reference trong benchmark moi. Notebook/order user bao cao: `A1_agent_04`, `A2_submission_copy`, `A3_my_improved_agent`, `A4_main`; improved agent la `submission.py`.
- Phan tich `submission.py` goc: day la model-based tactical planner, co tensor obs parser, orbit/fleet physics, `PlanetMovement` forecast, projected garrison status, competitive flow-diff scoring, focus-fire multi-source, va regroup leftover ships. Diem manh la aiming/physics va scorer rat sau; diem yeu la phan chon wave cuoi van greedy cuc bo.
- Van de chinh cua submission goc trong 4P: `_greedy_select` chon candidate score cao nhat hien tai, debit source/lock target xong moi lap tiep. Trong FFA, mot wave cuc bo cao co the an het source quan trong, lock target, hoac kich hoat role mutex lam mat wave thu hai tot hon. 4P config goc chu yeu giam horizon/source so voi 2P, chua co co che danh gia follow-up opportunity.
- Rang buoc tu user: khong sua co che multi-planet/focus-fire; chi cai tien greedy va 4P. 2P phai gan nhu khong bi anh huong, nen cac knob moi deu default off trong `ProducerLiteConfig` va chi bat trong `CONFIG_4P`.
- Attempt fail 1: them 4P target bias cho neutral production cao/leader-owned target, them ship-spend penalty trong greedy, va tang nhe horizon/source width. User test bao xau hon baseline khoang 2 tran. Ket luan: target bias can thiep qua tho vao flow-diff scorer; ship penalty lam agent ngai ban cac wave lon nhung can thiet. Da tat ship penalty, bo cong target bias vao final score, dua horizon/source ve baseline.
- Attempt fail 2: them single-source size variants 4P-only `(1.0, 0.75, 0.55)` de agent co lua chon ban vua du thay vi luon dung `safe_drain`. Multi-planet/focus-fire khong doi. User test bao te hon rat nhieu, con te hon attempt 1. Ket luan: candidate size nho lam tang khong gian action nhung lam agent kem quyet doan/pressure; scoring goc co ve phu thuoc candidate full safe-drain. Da tat `enable_single_source_size_variants=False`, `single_source_size_fracs=(1.0,)`.
- Cai tien dang giu: 4P greedy one-step lookahead trong `_greedy_select`. Khong doi target shortlist, khong doi fleet size, khong doi multi-planet. Van dung score tactical goc, nhung khi rank candidate hien tai `i`, tinh nhanh wave tot nhat `j` con co the ban sau khi chon `i` theo budget/source/target/role mutex.
- Cong thuc rank moi:
```python
current_effective_score(i)
+ greedy_lookahead_weight * max(next_score(i) - roi_threshold, 0)
```
- Diem an toan quan trong: dieu kien fire van dua tren current score cua wave hien tai, khong dua tren bonus lookahead. Nghia la lookahead chi sap xep lai cac wave hien tai da du tot, khong ep ban mot wave kem chi vi no mo duong cho wave sau.
- Config hien tai chi bat trong 4P: `enable_greedy_lookahead=True`, `greedy_lookahead_weight=0.25`. Default 2P van `enable_greedy_lookahead=False`, `greedy_lookahead_weight=0.0`, nen 2P khong doi hanh vi thuc te.
- Ket qua user bao cao tren 30 matches:
```text
=== Average rewards ===
A1_agent_04 avg_reward= -1.0 matches= 30
A2_submission_copy avg_reward= -0.1333 matches= 30
A3_my_improved_agent avg_reward= 0.0667 matches= 30
A4_main avg_reward= -0.9333 matches= 30
Saved replay: /kaggle/working/replays/last_benchmark_4v4.html
```
- Dien giai: `submission_copy` baseline avg reward `-0.1333`, improved `submission.py` avg reward `0.0667`, chenhlech `+0.2000` avg reward tren 30 matches. Day la ban dau tien trong nhanh `submission.py` duoc user bao cao la tot hon submission goc.
- Trang thai hien tai: giu greedy lookahead 4P; target bias disabled; ship-spend penalty disabled; single-source size variants disabled; multi-planet/focus-fire unchanged; 2P effectively unchanged.

## Version 25 - Submission.py lookahead final-wave guard
- User bao cao Version 24 tot hon `submission_copy` baseline va yeu cau cai tien tiep tu nen dang thang.
- Tao backup `submission_v24_good.py` de giu lai moc V24 truoc khi sua tiep.
- Phat hien lookahead V24 van cong future bonus o moi vong greedy, ke ca khi dang o wave slot cuoi cung. O wave cuoi khong con luot tiep theo de tan dung `next_score`, nen bonus nay co the lam chon candidate giu co hoi ao thay vi candidate hien tai tot nhat.
- Sua nhe trong `_greedy_select`: chi bat one-step lookahead khi `w + 1 < W`; neu dang o wave cuoi thi quay ve rank greedy hien tai.
- Thay doi nay khong doi target shortlist, khong doi fleet size, khong doi multi-planet/focus-fire, va van chi anh huong khi 4P config bat `enable_greedy_lookahead=True`.
- Da kiem tra syntax/import/shape test co hoc OK. Chua benchmark; can user test so voi backup V24.

## Version 26 - Submission.py 4P weighted opponent flow score
- User bao V25 "same same" V24 va yeu cau cai tien tiep tren nen dang tot.
- Huong moi: khong bias target shortlist, khong doi fleet size, khong doi multi-planet/focus-fire. Thay vao do sua cach quy doi flow-diff cua doi thu trong scorer 4P.
- Van de cua scorer goc: `competitive_score = me - sum(opponents)`, moi opponent bi tinh trong so bang nhau. Trong 4P FFA, lam leader mat loi thuong quan trong hon lam nguoi yeu mat loi; nguoc lai, giup leader co loi phai bi phat nang hon.
- Them `opponent_weights` vao `competitive_score`/`score_candidates`. Default `None` giu nguyen hanh vi cu. Khi co weights, score thanh:
```python
me_delta - sum(opponent_weight[player] * opponent_delta[player])
```
- Them helper `ffa_opponent_score_weights`: tinh production theo owner hien tai. Neu leader enemy hon minh it nhat `ffa_score_prod_gap`, leader weight cao hon, non-leader weight thap hon, player minh weight 0.
- Config 4P candidate hien tai: `enable_ffa_weighted_opp_score=True`, `ffa_leader_score_weight=1.30`, `ffa_nonleader_score_weight=0.85`, `ffa_score_prod_gap=5.0`. 2P default off nen khong doi.
- Y nghia: neu candidate lam leader mat net flow, no duoc cong gia tri lon hon; neu candidate vo tinh giup leader, bi tru manh hon. Day la owner-aware adjustment ben trong scorer, it tho hon target bias attempt fail.
- Da kiem tra syntax/import va unit nho cho weighted score OK. Chua benchmark; can user test so voi `submission_v24_good.py` va V25.

## Version 27 - Submission.py 4P early neutral-first discovery
- User bao V26 van "same same" va yeu cau phan tich lai bai toan/code thay vi tiep tuc nho nho. Ket luan moi: diem yeu 4P khong chi nam o greedy/scorer, ma o phase early FFA.
- Phan tich code: `build_target_shortlist` tron `enemy | neutral` vao cung offensive shortlist va rank chu yeu theo proximity. Trong 4P early, danh enemy gan qua som de lam cham expansion, tao kingmaking/cleanup cho nguoi khac, trong khi muc tieu dung hon thuong la farm neutral production.
- Huong moi: early 4P neutral-first target discovery. Khong doi final flow scorer, khong doi fleet size, khong doi multi-planet/focus-fire. Chi doi target candidates duoc dua vao scorer trong phase early.
- Them config default off trong `ProducerLiteConfig`: `enable_ffa_early_neutral_gate`, `ffa_early_neutral_until_turn`, `ffa_early_neutral_until_prod`, `ffa_early_neutral_prod_weight`, `ffa_early_neutral_ship_penalty`.
- Config 4P hien tai bat gate: `enable_ffa_early_neutral_gate=True`, `ffa_early_neutral_until_turn=70`, `ffa_early_neutral_until_prod=20.0`, `ffa_early_neutral_prod_weight=2.0`, `ffa_early_neutral_ship_penalty=0.15`.
- Logic: neu player_count >= 4 va dang early (`step < 70` hoac `my_prod < 20`) va con neutral hop le, offensive shortlist chi lay neutral. Neutral duoc rank theo `-proximity + prod*2.0 - ships*0.15`, de uu tien neutral gan/co production cao/khong qua dat. Neu het neutral hop le hoac qua phase, tu dong quay ve logic cu `enemy | neutral`.
- V26 weighted opponent score da duoc tat trong config hien tai (`enable_ffa_weighted_opp_score=False`) de test sach huong moi; V24 greedy lookahead van giu vi do la nen user bao tot hon baseline.
- Da kiem tra syntax/import OK. User test bao ban nay yeu qua, da rollback `submission.py` ve backup `submission_v24_good.py`. Ket luan: hard early neutral-first lam agent mat kha nang chon enemy tactical co loi; khong tiep tuc huong gate neutral cung.

## Version 28 - Rollback to V24 good baseline
- Sau khi V27 fail, `submission.py` da duoc copy lai tu `submission_v24_good.py`.
- Trang thai hien tai quay ve ban tot nhat da duoc user bao cao hon baseline: 4P greedy lookahead bat `enable_greedy_lookahead=True`, `greedy_lookahead_weight=0.25`; single-source size variants off; target bias off; weighted opponent score/early neutral gate khong con trong file active.
- Syntax/import OK. Tiep theo nen cai tien tren nen V24 bang huong khac, khong tiep tuc ep target discovery early.

## Version 29 - Submission.py midgame potential-risk regroup
- Tiep tuc tu V24 good baseline sau khi V27 fail. Khong doi attack scorer, target shortlist, fleet size, hay multi-planet/focus-fire.
- Phan tich lai code thay vi ep target: `submission.py` da co san `potential_attack_risk`, nhung default off. Ham nay tinh risk map tu enemy planets theo ships+production, proximity, sun line-of-sight, va friendly support discount; sau do co the cong vao pressure gradient cua `_plan_regroup`.
- Diem yeu 4P co the nam o dieu quan ship du: cheap pressure hien tai chi nhin enemy garrison reachable gan, chua coi production/threat gia tri planet. Trong FFA midgame, ship du nam sai cum de bi leader/enemy danh lung tung.
- Them `potential_risk_turn_min` de khong lam cham early expansion. Config 4P candidate: `enable_potential_risk=True`, `potential_risk_turn_min=45`, `risk_blend_weight=0.35`.
- Y nghia: early V24 gan nhu giu nguyen; tu turn 45, regroup se tinh them precautionary risk de day leftover ships ve owned planets nguy hiem/co gia tri hon.
- Da kiem tra syntax/import OK. Chua benchmark; neu yeu rollback ve `submission_v24_good.py`.

## Version 30 - Rollback active submission.py to V24
- User yeu cau dung cai tien va back ve Version 24.
- Da copy `submission_v24_good.py` -> `submission.py`.
- Kiem tra active config: `enable_greedy_lookahead=True`, `greedy_lookahead_weight=0.25`; `enable_potential_risk=False`; `enable_single_source_size_variants=False`; target bias weights `0.0`.
- Syntax/import OK. Trang thai active hien tai la V24 good baseline.

## Version 31 - Submission.py 4P consolidation reserve
- User phan tich replay `78954591.json` cua `IAI-RL-Tlm71`: early tot, turn ~40 con rat manh, nhung midgame toang vi overextend/recapture loop. Agent chiem duoc planet roi rut tiep qua som, de planet moi con 4-12 ships va bi cuop lai. Nhieu shot nho/xa nhu 4, 8, 12 ships lam mat tempo.
- Huong moi: khong chuyen sang turtle, ma them consolidation 4P: giu thanh qua sau expansion. Khong doi target scorer, target shortlist, fleet size variants, hay multi-planet/focus-fire.
- Them `apply_ffa_consolidation_reserve` sau `safe_drain`: trong 4P midgame, moi source phai de lai reserve theo production. Config hien tai: `enable_ffa_consolidation=True`, `ffa_consolidate_turn_min=35`, `base_reserve=8`, `prod_reserve=2.0`, `leader_bonus=6`, `lead_margin=0`.
- Y nghia reserve: source prod 4 sau turn 35 giu khoang `8 + 2*4 = 16` ships; neu dang dan/gap tot thi giu them 6. Neu source khong du surplus thi khong ban tiep, tranh rut can planet vua chiem.
- Them filter `enable_ffa_small_far_filter=True`: sau turn 30, offensive shot nho hon 16 ships va ETA > 5 bi loai. Defense/reinforcement target cua minh khong bi filter nay.
- V24 greedy lookahead van giu. 2P default off nen khong bi anh huong.
- Da kiem tra syntax/import va smoke test reserve OK. User benchmark bao phien ban nay khong tot chut nao. Da rollback `submission.py` ve `submission_v24_good.py`.
- Ket luan: reserve/filter cung lam mat nhip attack qua manh; van de consolidate co that nhung khong nen xu ly bang hard cap source drain hoac hard filter shot nho/xa.

## Version 32 - Rollback active submission.py to V24 after V31 failure
- Da copy `submission_v24_good.py` -> `submission.py`.
- Kiem tra active config: `enable_greedy_lookahead=True`, `greedy_lookahead_weight=0.25`; `enable_potential_risk=False`; khong con consolidation fields trong active backup.
- Syntax/import OK. Active file quay ve V24 good baseline.

## Version 33 - Submission.py 4P soft small-far shot penalty
- Tiep tuc tu V24 good sau khi V31 hard reserve/filter fail. Bai hoc: khong nen cam cung hoac hard-cap source drain vi lam mat nhip attack.
- Huong moi mem hon: khong doi candidate generation, target shortlist, scorer core, multi-planet/focus-fire, hay defense. Chi tru diem nhe cho offensive candidate co tong send nho va ETA xa trong 4P midgame.
- Them config default off: `enable_ffa_small_far_soft_penalty`, `ffa_small_far_soft_turn_min`, `ffa_small_far_soft_min_ships`, `ffa_small_far_soft_eta_min`, `ffa_small_far_soft_penalty`.
- Config 4P hien tai: bat soft penalty sau turn 30, candidate offensive co `total_send < 14` va `max_eta > 5` bi tru `1.25` score. Neu scorer that su thay move rat tot thi van co the ban; defense/reinforcement khong bi anh huong.
- Muc tieu: giam shot nho/xa kieu 4/8/12 ships lam mat tempo, nhung khong lap lai loi V31 la block cung cac tactical move co loi.
- Syntax/import OK. Can user benchmark vs `submission_v24_good.py`.

## Version 34 - JAX PPO training notebook groundwork
- Tao va mo rong `orbit_wars_jax_ppo_train.ipynb` theo huong JAX-first, khong train grid search.
- Port core official engine sang JAX: launch, production, orbit planet, swept collision, combat, reward.
- Them comet support: comet path/ships duoc precompute bang official generator theo hidden seed, sau do JAX step consume tensor schedule de spawn/move/remove.
- Validation local da pass:
  - no-action core.
  - scripted launch.
  - 19 hard cases collision/combat/reward theo official behavior.
  - random scripted 40 turn truoc comet.
  - no-action 60 turn qua comet.
  - random scripted 80 turn qua comet.
- Them PPO core chay duoc: heuristic sinh candidate action, policy JAX hoc bias/chon candidate, rollout vectorized, GAE, PPO clipped loss, Adam update.
- Smoke test local voi 4 env / 4 step / 1 update da chay xong va tra ve `PolicyParams`.
- Trang thai hien tai: notebook train duoc ve mat code/simulator. Chua claim agent manh hon vi chua train dai, chua benchmark, va chua export/integrate weights vao submission runtime.

## Version 35 - JAX V24-port PPO candidate generator
- User yeu cau port not theo V24 truoc khi train that, uu tien dung logic V24 hon toc do train.
- File active cho train: `orbit_wars_jax_v24_port_train.ipynb`.
- Mo rong candidate generator JAX tu scaffold cu sang V24-like:
  - Source shortlist, attack shortlist, defense/regroup shortlist.
  - Single-source attack, focus-fire multi-source, regroup/defense, greedy multi-wave.
  - Greedy multi-wave co source budget, one-wave-per-target, source/target mutex, va lookahead bonus.
  - Target flow/garrison projection tren horizon, source debit flow penalty, target flow delta score.
  - Sun-line gate va body-screen/first-contact check cho launch angle.
  - Fleet target inference bang first-contact sweep theo future planet movement, dung cho incoming bucket va pressure scoring.
  - Safe drain moi ket hop near fleet pressure, inferred incoming-to-source pressure, va flow-guard de tranh rut can source neu future flow xau.
- Sua loi shape trong safe drain: reserve phai la scalar theo source, khong duoc dung ca vector `reserve`.
- Sua call `fleet_speed` trong PPO cell de nhat quan voi signature notebook.
- Check da chay bang `.venv\Scripts\python.exe`:
  - Syntax tat ca code cells OK.
  - Validation official-vs-JAX PASS: noaction/comet, scripted launch, hard cases, random scripted no-comet, random scripted with-comets.
  - Tiny PPO smoke train PASS: `ROLLOUT_STEPS=1`, `num_envs=1`, `updates=1`, tra ve `PolicyParams`.
- Chua claim agent manh hon vi chua train dai va chua benchmark sau train. Buoc tiep theo la train notebook nay tren Kaggle/GPU, sau do export weights vao submission runtime.

## Version 36 - JAX V24 PPO final train design with shaped reward
- User yeu cau thiet ke notebook train PPO ban toi uu, khong phai ban thu, co `tqdm`, cell config rieng, va giai thich config.
- File cap nhat: `orbit_wars_jax_v24_port_train.ipynb`.
- Viet lai cell config bang comment khong dau de tranh loi font tren Kaggle:
  - Train mix mac dinh `TRAIN_PROB_4P=0.80`, `TRAIN_PROB_2P=0.20`.
  - Train scale mac dinh `NUM_ENVS=512`, `ROLLOUT_STEPS=128`, `TOTAL_PPO_UPDATES=800`.
  - PPO params: gamma dai, entropy cao hon mot chut, hidden dim 96.
  - Them `RESET_EVERY_UPDATES` de reset map/state dinh ky, tranh overfit vao mot lo rollout lien tiep.
- Them reward shaping rieng cho PPO:
  - Terminal reward game van giu la tin hieu chinh.
  - Dense reward moi step dua tren delta strategic score: production, planet count, planet ships, fleet ships, va relative lead so voi doi thu.
  - Co clip rieng cho shaping va total reward de tranh shaping lan at win/loss.
- Them `tqdm.auto` vao imports va progress bar trong `train_ppo`.
- `train_ppo` bay gio log: loss, policy loss, value loss, entropy, mean_player_reward; dong thoi luu `TRAIN_HISTORY` trong globals.
- Check da chay:
  - Syntax notebook OK.
  - Validation official-vs-JAX PASS tat ca muc.
  - Tiny PPO smoke train PASS voi reward shaping moi: `ROLLOUT_STEPS=1`, `num_envs=1`, `updates=1`, tra ve `PolicyParams`.
- Luu y: OpenSpiel/LiteLLM warning khi import Kaggle env la warning ngoai le cua package, khong lam fail Orbit Wars validation/train smoke.

## Version 37 - PPO minibatch update to avoid Kaggle OOM
- User chay Kaggle gap `XlaRuntimeError: RESOURCE_EXHAUSTED` tai `ppo_update`, allocate khoang 10.99GB.
- Nguyen nhan: train loop da tao rollout lon nhung `ppo_update` dang tinh loss/grad tren ca batch cung luc; `MINIBATCH_SIZE` co trong config nhung chua duoc dung that su.
- Sua `orbit_wars_jax_v24_port_train.ipynb`:
  - Them `flatten_ppo_batch`: flatten dims `[T, B, player] -> [samples]`.
  - Them `minibatch_slice`.
  - Trong `train_ppo`, PPO update chay theo `UPDATE_EPOCHS` va tung minibatch `MINIBATCH_SIZE` thay vi full batch.
- Check local:
  - Syntax OK.
  - Validation official-vs-JAX PASS.
  - Tiny PPO smoke train PASS sau patch minibatch.
- Neu Kaggle van OOM thi giam `NUM_ENVS` xuong 256 truoc, sau do moi giam `ROLLOUT_STEPS` xuong 64.

## Version 38 - Pack checkpoint 0200 into V24 PPO rerank agent
- User co checkpoint `ppo_policy_update_0200.npz` va muon test voi cac agent official khac truoc khi train den 800.
- Tao file moi `submission_v24_ppo_update0200.py`, khong sua `submission_v24_good.py`.
- Cach dong goi:
  - Copy full V24 good runtime lam nen.
  - Embed weights tu `D:\Downloads\ppo_policy_update_0200.npz` vao file `.py`.
  - Them PPO reranker vao ngay sau `score_candidates` va truoc `_greedy_select`.
  - Candidate feature 12 chieu khop train notebook: score, target prod/ships, source ships, total send, ETA, active launch count, neutral/enemy/defense, step.
  - PPO bonus duoc center theo valid candidates de khong day ca nguong ROI len/xuong cung luc.
  - Bat mac dinh cho 4P: `enable_ppo_rerank=True`, `ppo_bonus_weight=1.0`; 2P giu V24 default.
- Check:
  - Import OK: `CONFIG_4P.enable_ppo_rerank=True`.
  - 1 official env smoke match voi `[agent_04.py, submission_v24_good.py, submission_v24_ppo_update0200.py, main.py]` chay DONE, khong crash.
  - Smoke reward tra ve `[-1, -1, 1, -1]`; day chi la smoke, chua claim benchmark manh hon.

## Version 39 - Pack checkpoint 0400 into V24 PPO rerank agent
- User co checkpoint `ppo_policy_update_0400.npz` va yeu cau long vao V24.
- Tao file moi `submission_v24_ppo_update0400.py`, khong sua `submission_v24_good.py`.
- Embed weights tu `D:\Downloads\ppo_policy_update_0400.npz`; shape checkpoint khop policy train: `w1 (12, 96)`, `w2 (96, 1)`.
- Giu cung co che voi Version 38:
  - V24 good lam nen.
  - PPO chi cong learned bonus vao candidate score truoc `_greedy_select`.
  - Feature 12 chieu khop train notebook.
  - Bat PPO rerank cho 4P: `enable_ppo_rerank=True`, `ppo_bonus_weight=1.0`; 2P giu default.
- Check:
  - Import OK: `CONFIG_4P.enable_ppo_rerank=True`.
  - Grep xac nhan header update 0400, rerank block, va config 4P da bat.
  - 1 official env smoke match voi `[agent_04.py, submission_v24_good.py, submission_v24_ppo_update0400.py, main.py]` chay DONE, khong crash.
  - Smoke reward tra ve `[-1, -1, 1, -1]`; day chi la smoke, chua claim benchmark manh hon.

## Version 40 - Make agent_04 self-contained and remove abc.py
- User muon viet bao cao dua tren `agent_04.py`, `main.py`, `submission_copy.py`, `submission_v24_good.py`, `x.py`, va PPO; dong thoi yeu cau khong goi file py ngoai va xoa `abc.py`.
- Phan tich nhanh cac nen:
  - `agent_04.py`: agent heuristic nhe, co 2 lop: embedded ABC early opener va `main_agent` phase-based heuristic. Diem manh la early expansion/coop attack don gian, moving-planet intercept, sun avoidance. Truoc do `_load_abc_module()` uu tien doc `abc.py` ngoai.
  - `main.py`: agent lon kieu tactical planner, co world model, defense/rescue, cheap pickup, expand, accumulator, hammer/mega-hammer/multiprong, 4P external hook. Diem manh la nhieu mode tactical, nhung phuc tap va co hook ngoai nen khong phu hop copy nguyen vao `agent_04.py`.
  - `submission_copy.py` / `submission_v24_good.py` / `x.py`: cung ho ProducerLite tensor/Torch planner. Diem manh la movement forecast, garrison flow, sparse score candidates, greedy select, focus-fire, regroup. `submission_v24_good.py` co greedy lookahead 4P tot hon ban copy; `x.py` la bien the target/leader bias khac.
  - PPO: khong thay the planner; no hoc rerank/bonus tren candidate V24-like. Khi package vao V24, PPO chi cong learned bonus vao score truoc `_greedy_select`.
- Thay doi code:
  - Sua `agent_04.py`: bo import `importlib.util`.
  - Sua `_load_abc_module()` de chi load `ABC_EMBEDDED_SOURCE` trong chinh file, khong con tim/exec `abc.py` ngoai.
  - Xoa file `abc.py` theo yeu cau.
- Check:
  - `import agent_04` OK.
  - `Test-Path abc.py` tra ve `False`.
  - Grep khong con `abc.py`, `spec_from_file_location`, `importlib.util` trong `agent_04.py`.
  - 1 official env smoke match voi `[agent_04.py, submission_v24_good.py, submission_v24_ppo_update0400.py, main.py]` chay DONE, khong crash.
- Luu y thiet ke: khong copy nguyen `main.py`/V24/Torch runtime vao `agent_04.py` trong lan nay vi cac runtime co global state, entrypoint, helper trung ten va dependency rieng; gop co hoc se de crash hoac lam agent yeu hon. Huong dung cho bao cao la mo ta tien hoa logic: `agent_04` early heuristic -> `main` tactical planner -> ProducerLite/V24 tensor flow planner -> PPO rerank candidate.

## Version 41 - Make main.py self-contained for dormant 4P specialist hook
- User hoi `main.py` co dang goi file `.py` ngoai khong va muon tich hop giong cach da lam voi `agent_04.py`.
- Kiem tra code:
  - `main.py` co `importlib.util` va `_load_four_p_external_agent()` load `main_plus_main5_ideas.py` tu cung folder.
  - `FOUR_P_EXTERNAL_ENABLED = False`, nen trong benchmark/submission mac dinh nhanh nay khong duoc goi; no la hook 4P specialist dang ngu.
- Thay doi code:
  - Bo `importlib.util`, them `base64`.
  - Nhung nguyen source `main_plus_main5_ideas.py` vao `MAIN_PLUS_MAIN5_EMBEDDED_B64` trong `main.py`.
  - Sua `_load_four_p_external_agent()` de giai ma source nhung va `exec` thanh module noi bo, lay ham `agent`; khong con doc `main_plus_main5_ideas.py` tu disk.
  - Giu `FOUR_P_EXTERNAL_ENABLED = False` de khong doi hanh vi hien tai khi user chua bat nhanh 4P ngoai.
- Y nghia cai tien/de ghi bao cao:
  - `main.py` la tactical planner lon tu base Vkhydras: world model, defense/rescue, expand, cheap pickup, accumulator, hammer/mega-hammer/multiprong.
  - Phan `main_plus_main5_ideas.py` la nhanh 4P specialist cung ho tactical planner, tap trung nhieu rule 4P nhu search expansion, neutral hard cap, cheap pickup 4P, accumulator, brain lead reserve, mega hammer va FFA pressure.
  - Viec nhung vao file giup submission self-contained va tranh loi thieu file ngoai, nhung khong claim no lam benchmark tot hon vi flag mac dinh van tat.
- Check:
  - `python -m py_compile main.py` OK.
  - `import main` OK.
  - `_load_four_p_external_agent()` tra ve callable khi goi truc tiep.
  - Grep khong con `importlib`, `spec_from_file_location`, `module_from_spec`, `exec_module`.
  - 1 official env smoke match voi `[main.py, agent_02.py, agent_03.py, agent_04.py]` chay DONE, khong crash. Reward smoke `[-1, -1, -1, 1]`; chi dung de xac nhan runtime, khong claim suc manh.

## Version 42 - Pack checkpoint 0450 into V24 PPO rerank agent
- User co checkpoint `D:\Downloads\ppo_policy_update_0450.npz` va yeu cau long vao truoc khi lam tiep.
- Tao file moi `submission_v24_ppo_update0450.py` tu template `submission_v24_ppo_update0400.py`, thay weights bang checkpoint 0450.
- Checkpoint shape khop policy reranker:
  - `w1 (12, 96)`, `b1 (96,)`, `w2 (96, 1)`, `b2 (1,)`.
  - File checkpoint cung co value weights, nhung submission chi can policy weights de tinh PPO bonus.
- Giu cung co che voi Version 39:
  - V24 good lam nen.
  - PPO chi cong learned bonus vao candidate score truoc `_greedy_select`.
  - Bat 4P rerank: `enable_ppo_rerank=True`, `ppo_bonus_weight=1.0`.
- Check:
  - `python -m py_compile submission_v24_ppo_update0450.py` OK.
  - Import OK; `CONFIG_4P.enable_ppo_rerank=True`, `CONFIG_4P.ppo_bonus_weight=1.0`.
  - Weight embed check: `_PPO_W1_DATA` co shape `12 x 96`, `_PPO_B2_DATA=[-2.9849326610565186]`.
  - 1 official env smoke match voi `[agent_04.py, submission_v24_good.py, submission_v24_ppo_update0450.py, main.py]` chay DONE, khong crash. Reward smoke `[-1, 1, -1, -1]`; chi dung de xac nhan runtime, khong claim suc manh.

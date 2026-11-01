# orbit_sim 多体引力 API 与交接手册

本文对应当前 `src/orbit_sim/` 的实际代码，面向初学者和后续捕获实验开发者。函数名、参数名与结果字段按代码原样列出；历史拼写也保留。本文中的 Python 代码块均可独立运行，不需要先执行其他代码块。

**先记住职责：dynamics 算加速度，integrators 推进一步，simulate 重复推进并保存历史，diagnostics 计算检查量。**

## 目录

1. [阅读路线与旧/新接口对照](#overview)
2. [系统状态、shape 与 active/test 规则](#state)
3. [完整入门示例：地月与航天器](#quickstart)
4. [dynamics.py API](#dynamics)
5. [integrators.py API](#integrators)
6. [simulate.py API：当前状态与历史](#simulate)
7. [diagnostics.py API](#diagnostics)
8. [circular_orbit.py：旧中心模型工具](#circular)
9. [内部辅助函数与调用框架](#internals)
10. [切换系统、参考系与旧代码迁移](#migration)
11. [最小捕获实验模板](#capture)
12. [时间步长、收敛与验证](#validation)
13. [常见错误与英文词汇](#faq)

<a id="overview"></a>
## 1. 阅读路线与旧/新接口对照

第一次使用时读第 2、3、6 节即可建立系统、运行和读取轨迹；理解实现时读第 4、5、9 节；设计捕获实验时读第 10–12 节。

### 1.1 代码位置与职责

| 文件 | 解决的问题 | 不负责的事情 |
| --- | --- | --- |
| `src/orbit_sim/dynamics.py` | 给定当前位置，计算加速度 | 不更新位置、速度，不保存轨迹 |
| `src/orbit_sim/integrators.py` | 给定状态、步长和加速度函数，推进一次 | 不配置天体名单，不累计完整历史 |
| `src/orbit_sim/simulate.py` | 重复调用积分器，保存状态和诊断历史 | 不判断是否捕获，不检测碰撞事件 |
| `src/orbit_sim/diagnostics.py` | 计算能量、动量、角动量、质心 | 不修改状态，不决定积分步长 |
| `src/orbit_sim/circular_orbit.py` | 旧固定 Earth 模型的圆轨道初值与解析解 | 不提供通用多体圆轨道解 |
| `experiments/lunar/003_nbody_validation/main.py` | 验证理想地月有限质量二体和 test 航天器 | 不是捕获成功案例，也不是真实星历 |

### 1.2 旧接口、新接口、扩展接口

“旧”表示在多体改动前已有；“扩展”表示在原函数上加入能力，旧调用保留；“新增”表示本次新加入。

| 状态 | 模块 | 函数/名称 | 当前定位 |
| --- | --- | --- | --- |
| 旧，保留 | dynamics | `gravitational_acceleration` | 单目标、固定 Earth 质量 |
| 新增 | dynamics | `gravitational_acceleration_from_sources` | 单目标、任意一组源 |
| 新增 | dynamics | `nbody_accelerations` | 同一快照全部目标的加速度 |
| 新增 | dynamics | `make_nbody_acceleration` | 制作记住质量/掩码的回调 |
| 扩展 | integrators | `eular`、`Symplectic_Euler`、`Velocity_Verlet_KDK`、`Velocity_Verlet_DKD` | 增加 `acceleration_function`，支持 `(3,)` / `(N,3)` |
| 新增别名 | integrators | `euler` | 与 `eular` 是同一个函数，修正英文拼写方便新代码使用 |
| 旧，保留 | simulate | `simulate_orbit`、`simulate_orbit_with_multiple_integrators` | 旧单目标固定 Earth 模拟 |
| 新增 | simulate | `simulate_nbody` | 多体同步推进与历史 |
| 旧，保留 | diagnostics | `specific_energy`、`specific_angular_momentum` | 旧比能量与比角动量 |
| 新增 | diagnostics | `total_energy`、`total_linear_momentum`、`total_angular_momentum`、`center_of_mass` | 所选有限质量子系统的总量 |
| 旧，未改 | circular_orbit | `circular_orbit_velocity`、`circular_orbit_period`、`analyical_circular_trajectory` | 固定原点 Earth 圆轨道工具 |
| 新增，内部 | dynamics / integrators / diagnostics | `_system_arrays` / `_prepare` / `_selected` | 输入整理、校验和选择，不是用户状态类 |

历史 `experiments/lunar/001_circular_orbit_convergence` / `experiments/lunar/002_circular_orbit_long_term` 保持原文件，不把它们改写成 N 体实验。不要重命名旧 `eular`、`analyical_circular_trajectory` 或旧结果字段 `anguluar_momentum`，这些历史拼写参与兼容。

### 1.3 当前适用范围

- 牛顿点质量引力；位置、速度在同一惯性坐标系中。
- 天体数量由数组长度决定，没有写死为 Earth/Moon，也没有“中心天体”参数。
- 不含软化、有限半径碰撞、推力、非球形引力、耗散或星历驱动。
- 当前加速度回调只接受位置，不接受时间或速度。
- 引力求值约为 O(N²)，保存完整状态历史约需 O(steps × N) 内存；任意 N 不等于任意规模都高效。

<a id="state"></a>
## 2. 系统状态、shape 与 active/test 规则

### 2.1 一个时刻的状态

| 数组 | 内容 | shape | 单位 |
| --- | --- | --- | --- |
| `positions`，下文也写 `R` 或内部 `r` | 当前全部天体位置 | `(N,3)` | m |
| `velocities`，也写 `V` 或内部 `v` | 当前全部天体速度 | `(N,3)` | m/s |
| `masses`，也写 `m` | 每个天体的质量 | `(N,)` | kg |
| `gravitating`，内部也写 `mask` | 每行是否作为引力源 | `(N,)`，布尔 | 无 |
| `accelerations`，也写 `A` | 当前全部加速度 | `(N,3)` | m/s² |

`shape` 是数组形状。`(3,)` 是一个三分量向量；`(N,3)` 是 N 行、每行 xyz。`N=1` 的多体状态仍应写成 `[[x,y,z]]`，shape 是 `(1,3)`，而不是 `(3,)`。

数组行号是天体身份：位置第 i 行、速度第 i 行、质量第 i 项、掩码第 i 项必须对应同一个天体。可以另存 `names`，但函数不根据名字判断 Earth 或 Moon。

### 2.2 哪些天体产生引力？

| 配置 | 接受其他源的引力 | 对其他目标产生引力 | 是否传播位置/速度 |
| --- | --- | --- | --- |
| `True`、正质量：active body | 是 | 是 | 是 |
| `False`：test body | 是 | 否 | 是 |
| 零质量 | 是 | 贡献为零，即使 mask=True | 是 |

`False` 不表示静止，也不是碰撞屏蔽。test 可以保留真实质量，例如 1000 kg；该质量不会改变自身加速度，也不反馈到其他天体。`gravitating=None` 默认全 True，**包含航天器时应显式给出掩码**。

标准地月与航天器：`[True, True, False]`。只考虑地球源：`[True, False, False]`，此时月球也成为只受力的试验体，模型物理含义已改变。

系统级函数通过行索引排除自身引力；不同目标与正质量源重合会报错。低层单目标函数不认识行身份，调用者需自行排除自身。空源集合返回零加速度；两个都不作为源的 test 重合本身不引发引力奇点。

### 2.3 当前位置与位置历史是两种数据

```text
传入 positions / 当前 r       (N,3)         一个时刻、全部天体
内部 history['positions']    Python 列表    多个快照，尚在收集
返回 data['positions']       (T,N,3)       多个时刻、全部天体
T = steps + 1
```

旧 `simulate_orbit` 中 `positions` 是单天体历史；新 `simulate_nbody` 的输入 `positions` 是当前快照，返回字典中的同名键才是历史。不要仅根据变量名称判断，应看它在什么函数中以及 shape。

<a id="quickstart"></a>
## 3. 完整入门示例：地月与航天器

以下代码独立运行。它先建立质心在原点、总动量近零的理想地月圆轨道，再在地球附近加入 test 航天器。航天器初速度仅近似为地球圆轨道速度；加入月球后不是精确圆轨道。这是接口演示，不是捕获案例。

```python
import numpy as np
from orbit_sim.dynamics import G, M_EARTH, nbody_accelerations
from orbit_sim.integrators import Velocity_Verlet_KDK
from orbit_sim.simulate import simulate_nbody

names = ["Earth", "Moon", "spacecraft"]
masses = np.array([M_EARTH, 7.342e22, 1000.0])
gravitating = np.array([True, True, False])
distance = 3.844e8
active_mass = masses[0] + masses[1]
relative_speed = np.sqrt(G * active_mass / distance)

positions = np.array([
    [-masses[1] / active_mass * distance, 0.0, 0.0],
    [ masses[0] / active_mass * distance, 0.0, 0.0],
    [0.0, 0.0, 0.0],
])
velocities = np.array([
    [0.0, -masses[1] / active_mass * relative_speed, 0.0],
    [0.0,  masses[0] / active_mass * relative_speed, 0.0],
    [0.0, 0.0, 0.0],
])
# 地心相对初值转换为同一个惯性坐标系中的初值。
positions[2] = positions[0] + [1.0e7, 0.0, 0.0]
velocities[2] = velocities[0] + [0.0, np.sqrt(G * masses[0] / 1.0e7), 0.0]

accelerations = nbody_accelerations(positions, masses, gravitating)
assert accelerations.shape == (3, 3)

data = simulate_nbody(
    positions=positions,
    velocities=velocities,
    masses=masses,
    dt=20.0,
    steps=100,
    integrator=Velocity_Verlet_KDK,
    gravitating=gravitating,
)
assert data["positions"].shape == (101, 3, 3)
print("final time (s):", data["times"][-1])
print("spacecraft final position (m):", data["positions"][-1, 2])
```

这个例子中的 `nbody_accelerations` 只用于查看初始加速度。模拟每一步需要的加速度由 `simulate_nbody` 内部自动重新计算，不是重复使用最初的 A。

<a id="dynamics"></a>
## 4. dynamics.py API

### 4.1 `[旧] gravitational_acceleration`

**用途：** 单个目标受固定质量 Earth 的加速度。Earth 位置可由参数指定，但函数不会自行推进 Earth。

```text
gravitational_acceleration(point_position, earth_position=np.array([0.,0.,0.]))
```

| 参数 | 要求 | 默认 |
| --- | --- | --- |
| `point_position` | 单目标位置 `(3,)`，旧代码建议传 NumPy 数组，m | 必填 |
| `earth_position` | Earth 位置 `(3,)`，m | 原点 |

**返回：** `(3,)` 加速度，m/s²。质量始终使用 `M_EARTH`，没有质量参数。目标与 Earth 重合抛出 `ValueError`。旧函数没有新接口那样完整的 shape/有限值校验。

**机制：** 计算相对位置 r，再返回 `-G*M_EARTH*r/|r|³`。不能仅改 `earth_position` 就把它当成 Moon 引力。

```python
import numpy as np
from orbit_sim.dynamics import gravitational_acceleration

r = np.array([1.0e7, 0.0, 0.0])
a = gravitational_acceleration(point_position=r)
np.testing.assert_allclose(a, [-3.98589196, 0.0, 0.0])
```

### 4.2 `[新增] gravitational_acceleration_from_sources`

**用途：** 在一个指定目标位置，叠加任意一组点质量源的引力。可用于只查看航天器某个位置的受力。

```text
gravitational_acceleration_from_sources(target_position, source_positions, source_masses)
```

| 参数 | 类型/shape | 含义 |
| --- | --- | --- |
| `target_position` | 数值数组 `(3,)` | 目标位置，m |
| `source_positions` | 数值数组 `(M,3)` | M 个源的位置，m |
| `source_masses` | 非负数值数组 `(M,)` | 同序源质量，kg |

**返回：** `(3,)` 加速度，m/s²；不是力 N。目标质量不参与 a 的计算。

**实现机制：**

1. 校验目标和源；筛掉零质量源。
2. 对每个源求 `displacement = source_position - target_position`，方向指向源。
3. 求 `radius = norm(displacement)`。
4. 每个源贡献 `G*mass*displacement/radius³`，最后沿源的维度求和。

\[
\mathbf a=\sum_j Gm_j\frac{\mathbf r_j-\mathbf r_{target}}{|\mathbf r_j-\mathbf r_{target}|^3}
\]

**错误与边界：** 非法 shape、非有限数、负质量、与正质量源重合会报 `ValueError`。空源要写成 `np.empty((0,3))`，不是形状 `(0,)` 的 `[]`。该函数没有掩码参数，也不自动排除目标自身；传入前选择源。

```python
import numpy as np
from orbit_sim.dynamics import M_EARTH, gravitational_acceleration_from_sources

source_positions = np.array([[0.0, 0.0, 0.0], [3.844e8, 0.0, 0.0]])
source_masses = np.array([M_EARTH, 7.342e22])
a = gravitational_acceleration_from_sources(
    target_position=np.array([1.0e7, 0.0, 0.0]),
    source_positions=source_positions,
    source_masses=source_masses,
)
assert a.shape == (3,)
assert a[0] < 0  # 此处地球吸引占主要部分。
zero = gravitational_acceleration_from_sources([1.0, 0.0, 0.0], np.empty((0,3)), [])
np.testing.assert_array_equal(zero, np.zeros(3))
```

### 4.3 `[新增] nbody_accelerations`

**用途：** 给同一时刻的整组系统位置，一次返回全部目标加速度。

```text
nbody_accelerations(positions, masses, gravitating=None)
```

| 参数 | shape | 说明 |
| --- | --- | --- |
| `positions` | `(N,3)` | 一个时刻的全部位置 |
| `masses` | `(N,)` | 非负质量 |
| `gravitating` | `(N,)` 布尔或 None | 默认全部为源；不能用整数 `[1,0]` 代替布尔 |

**返回：** `(N,3)` 加速度，行顺序与输入一致。所有行都是受力目标，包括 mask=False 的行。

**实现机制：**

```text
对每个目标行 i：
    sources = gravitating AND (行号不是 i)
    选择 positions[sources] 与 masses[sources]
    调用 gravitational_acceleration_from_sources
    把结果放入 A[i]
返回 A
```

源码中的 `mask & (indices != i)` 是两条条件逐项“并且”。只循环计算加速度，不更新位置；因此循环不会混入不同时间的状态。

**错误：** 输入规则由 `_system_arrays` 检查；不同目标与所选正质量源重合由低层函数报错。单个孤立天体自动排除自己，返回零加速度。

```python
import numpy as np
from orbit_sim.dynamics import M_EARTH, nbody_accelerations

R = np.array([[0.,0.,0.], [3.844e8,0.,0.], [1.0e7,0.,0.]])
m = np.array([M_EARTH, 7.342e22, 1000.])
mask = np.array([True, True, False])
A = nbody_accelerations(R, m, mask)
assert A[0,0] > 0  # Earth 朝 Moon。
assert A[1,0] < 0  # Moon 朝 Earth。
np.testing.assert_allclose(m[0]*A[0], -m[1]*A[1])  # 第三定律比较的是力。
A_without_spacecraft = nbody_accelerations(R[:2], m[:2])
np.testing.assert_array_equal(A[:2], A_without_spacecraft)
```

### 4.4 `[新增] make_nbody_acceleration`

**用途：** 制作一个已配置质量/掩码的函数，供积分器反复调用。它返回函数，不返回当下的加速度。

```text
make_nbody_acceleration(masses, gravitating=None) -> acceleration
acceleration(positions) -> accelerations
```

**参数：** `masses (N,)`、`gravitating (N,)` 或 None，规则同上。

**返回：** 一个 callable（可调用对象），签名为 `acceleration(positions)`，输入/输出均 `(N,3)`。

**机制：** 先校验质量/掩码，然后复制两者，定义内部函数：

```text
def acceleration(positions):
    return nbody_accelerations(positions, 已复制的质量, 已复制的掩码)
```

这叫 closure（闭包）：函数记住外部配置。位置不被固定，积分器每次传入当前阶段的新位置。修改原始质量或掩码不会更新已有回调；增减天体或改配置后重新制作回调。

```python
import numpy as np
from orbit_sim.dynamics import make_nbody_acceleration, nbody_accelerations

R = np.array([[0.,0.,0.], [2.,0.,0.]])
m = np.array([3.,4.])
mask = np.array([True,False])
acceleration = make_nbody_acceleration(m, mask)
expected = nbody_accelerations(R, m, mask)
np.testing.assert_allclose(acceleration(R), expected)
m[:] = 0.0
mask[:] = False
np.testing.assert_allclose(acceleration(R), expected)  # 配置已复制。
```

<a id="integrators"></a>
## 5. integrators.py API

### 5.1 `[扩展] 四种积分器的共同约定

```text
eular(position, velocity, dt, acceleration_function=None)
euler(position, velocity, dt, acceleration_function=None)  # 新别名
Symplectic_Euler(position, velocity, dt, acceleration_function=None)
Velocity_Verlet_KDK(position, velocity, dt, acceleration_function=None)
Velocity_Verlet_DKD(position, velocity, dt, acceleration_function=None)
```

| 参数 | 要求 |
| --- | --- |
| `position` | 当前一个目标 `(3,)` 或整组系统 `(N,3)`，m；参数名仍为单数 |
| `velocity` | 与 position 完全相同的 shape，m/s |
| `dt` | 有限标量，s；直接积分器允许 0 和负值 |
| `acceleration_function` | `a = callback(position)`，输出必须与 position 同 shape 且有限；默认 None |

**返回：** 元组 `(new_position, new_velocity)`，shape 与输入一致。用户输入不被这些内置积分器就地更新；调用者要接住返回值。

**默认行为与兼容：** 没传回调时仍用原点 Earth 引力。对 `(N,3)` 默认是每行独立受固定 Earth 吸引，不是行与行相互作用。做 N 体时应传 `make_nbody_acceleration(...)` 返回的函数，或者使用自动配置它的 `simulate_nbody`。

**回调规则：** 回调只依赖本阶段位置。积分器给它一个只读的完整快照，防止修改状态；不得捕获并改动别处的状态或用未更新的旧 positions 替代参数。不要把 `acceleration_function=acceleration(R)` 传进去，这传的是计算结果；正确形式是 `acceleration_function=acceleration`。

**错误：** `_prepare` 校验状态 shape、有限值和 dt；每次力求值校验回调结果。源重合错误会继续向外抛出。当前接口不提供时间/速度参数；涉及推力、拖曳或时变星历时需先扩展传播设计。

### 5.2 `[扩展] eular / [新增别名] euler

**用途与机制：** 显式 Euler，一阶；一次求加速度，都使用旧状态。

\[
R_{n+1}=R_n+V_n dt,\qquad V_{n+1}=V_n+A(R_n)dt
\]

适合学习和误差对照。轨道长期能量漂移通常明显。

```python
import numpy as np
from orbit_sim.integrators import eular, euler

assert euler is eular
R = np.array([1.,0.,0.])
V = np.array([0.,1.,0.])
Rn, Vn = euler(R, V, 0.1, acceleration_function=lambda r: -r)
np.testing.assert_allclose(Rn, [1.,0.1,0.])
np.testing.assert_allclose(Vn, [-0.1,1.,0.])
```

这里的 `lambda r: -r` 是一个简短函数，返回 `-r`。它是用于学习回调的简谐振子模型（隐含频率为 1 s⁻¹），不是引力模型。

### 5.3 `[扩展] Symplectic_Euler

**用途与机制：** 辛 Euler，一阶；先更新速度，再用新速度更新位置，一次求加速度。

\[
V_{n+1}=V_n+A(R_n)dt,\qquad R_{n+1}=R_n+V_{n+1}dt
\]

```python
import numpy as np
from orbit_sim.integrators import Symplectic_Euler

R = np.array([1.,0.,0.])
V = np.array([0.,1.,0.])
Rn, Vn = Symplectic_Euler(R, V, 0.1, acceleration_function=lambda r: -r)
np.testing.assert_allclose(Rn, [0.99,0.1,0.])
np.testing.assert_allclose(Vn, [-0.1,1.,0.])
```

### 5.4 `[扩展] Velocity_Verlet_KDK

**用途：** 当前多体实验的常用选择，二阶。Kick 更新速度；Drift 更新位置。

**机制：** 所有天体一起半 kick → 全 drift → 半 kick：

\[
V_{half}=V_n+\tfrac12dt A(R_n),\quad
R_{n+1}=R_n+dt V_{half},\quad
V_{n+1}=V_{half}+\tfrac12dt A(R_{n+1})
\]

源码用等价写法 `R_new = R + V*dt + 0.5*A*dt**2`。每步两次加速度求值，分别使用完整旧位置和完整新位置，不把已更新的某一行与旧行混用。

```python
import numpy as np
from orbit_sim.dynamics import M_EARTH, make_nbody_acceleration
from orbit_sim.integrators import Velocity_Verlet_KDK

R = np.array([[0.,0.,0.], [3.844e8,0.,0.]])
V = np.array([[0.,0.,0.], [0.,1000.,0.]])
a = make_nbody_acceleration([M_EARTH, 7.342e22])
Rn, Vn = Velocity_Verlet_KDK(R, V, 20., acceleration_function=a)
assert Rn.shape == Vn.shape == (2,3)
np.testing.assert_array_equal(R[0], [0.,0.,0.])  # 原输入保留。
assert Rn[0,0] > 0  # Earth 已受 Moon 吸引。
```

### 5.5 `[扩展] Velocity_Verlet_DKD

**用途：** 二阶，可与 KDK 对照；有限步长下结果不必完全一致。

**机制：** 全部位置半 drift → 全部速度全 kick → 全部位置半 drift：

\[
R_{half}=R_n+\tfrac12dt V_n,\quad
V_{n+1}=V_n+dt A(R_{half}),\quad
R_{n+1}=R_{half}+\tfrac12dt V_{n+1}
\]

每步一次加速度求值，使用全部天体的统一半步位置。

```python
import numpy as np
from orbit_sim.dynamics import M_EARTH, make_nbody_acceleration
from orbit_sim.integrators import Velocity_Verlet_DKD

R = np.array([[0.,0.,0.], [3.844e8,0.,0.]])
V = np.array([[0.,0.,0.], [0.,1000.,0.]])
a = make_nbody_acceleration([M_EARTH, 7.342e22])
Rn, Vn = Velocity_Verlet_DKD(R, V, 20., acceleration_function=a)
assert Rn.shape == Vn.shape == (2,3)
assert Rn[0,0] > 0
```

<a id="simulate"></a>
## 6. simulate.py API：当前状态与历史

### 6.1 `[旧] simulate_orbit`

```text
simulate_orbit(position, velocity, dt, steps, integrator)
```

**用途：** 单个目标的旧固定 Earth 模拟。`position/velocity` 为 `(3,)` NumPy 数组，`dt` 单位 s，`steps` 为步数，`integrator` 是接受三个旧参数的函数。

**机制：** 复制初值 → 保存初始比能量/比角动量 → 每次调用积分器 → 保存返回状态 → 组成数组返回。它不传通用加速度参数，诊断仍固定为 Earth；不要用它承接新的有限质量系统。

| 返回键 | shape | 含义 |
| --- | --- | --- |
| `positions` | `(T,3)` | 一个目标的历史位置 |
| `velocities` | `(T,3)` | 历史速度 |
| `energy` | `(T,)` | 旧 Earth 比能量，J/kg |
| `anguluar_momentum` | `(T,3)` | 比角动量，m²/s；保留历史拼写 |

`T=steps+1`。旧接口不返回 `times`，自行用 `np.arange(T)*dt` 建立。它没有 `simulate_nbody` 的严格步数/步长校验，旧行为保持不变。

```python
import numpy as np
from orbit_sim.integrators import Velocity_Verlet_KDK
from orbit_sim.simulate import simulate_orbit

r = np.array([1.0e7,0.,0.])
v = np.array([0.,1000.,0.])
data = simulate_orbit(r, v, 10., 2, Velocity_Verlet_KDK)
assert data["positions"].shape == (3,3)
assert data["anguluar_momentum"].shape == (3,3)
times = np.arange(3)*10.
np.testing.assert_array_equal(times, [0.,10.,20.])
```

### 6.2 `[旧] simulate_orbit_with_multiple_integrators`

```text
simulate_orbit_with_multiple_integrators(position, velocity, dt, steps, integrators)
```

**用途：** 对同一个旧中心模型初值分别运行不同积分器。这里的 multiple integrators 是“多个积分器”，不是“多个天体”。

`integrators` 的实际类型是名称到函数的**字典**（旧源码 docstring 曾写 list，应以代码 `.items()` 为准）。其他输入同 6.1。

**返回：** `results[method_name]` 是该方法的旧 `simulate_orbit` 结果字典。各方法从相同初值独立运行，不是依次接着上一方法的终态。

```python
import numpy as np
from orbit_sim.integrators import Velocity_Verlet_KDK, Velocity_Verlet_DKD
from orbit_sim.simulate import simulate_orbit_with_multiple_integrators

results = simulate_orbit_with_multiple_integrators(
    np.array([1.0e7,0.,0.]), np.array([0.,1000.,0.]), 10., 2,
    {"KDK": Velocity_Verlet_KDK, "DKD": Velocity_Verlet_DKD},
)
assert results["KDK"]["positions"].shape == (3,3)
assert results["DKD"]["positions"].shape == (3,3)
```

### 6.3 `[新增] simulate_nbody`

```text
simulate_nbody(positions, velocities, masses, dt, steps, integrator,
               gravitating=None, acceleration_function=None)
```

**用途：** 整组系统同步推进，并记录初值、每个完整步的状态和诊断量。固定数量、固定质量、固定掩码的一段传播。

| 参数 | 类型/shape | 默认与限制 |
| --- | --- | --- |
| `positions` | `(N,3)` | 初始位置，m |
| `velocities` | `(N,3)` | 初始速度，m/s |
| `masses` | `(N,)` | 非负质量，kg |
| `dt` | 标量 | 有限且严格 > 0，s；与单步积分器约定不同 |
| `steps` | Python / NumPy 整数 | >= 0，拒绝 bool 和浮点数 |
| `integrator` | 函数 | 接受 `(R,V,dt,acceleration_function=...)`，返回同 shape 状态 |
| `gravitating` | 布尔 `(N,)` 或 None | 默认全部 True，同时用于引力与诊断选择 |
| `acceleration_function` | 函数或 None | 默认按 masses/mask 自动建立 N 体回调 |

**返回字典：** `T=steps+1`，初始状态占索引 0；最终时间 `steps*dt`。

| 键 | shape | 单位与说明 |
| --- | --- | --- |
| `positions` | `(T,N,3)` | m，位置历史 |
| `velocities` | `(T,N,3)` | m/s，速度历史 |
| `times` | `(T,)` | s，`0, dt, 2dt, ...` |
| `total_energy` | `(T,)` | J，只统计 mask 选中的 active 部分 |
| `total_linear_momentum` | `(T,3)` | kg·m/s |
| `total_angular_momentum` | `(T,3)` | kg·m²/s，关于惯性原点 |
| `center_of_mass` | `(T,3)` | m，所选质心 |

与旧结果的区别：旧 `(T,3)` 增加了天体维度，成为 `(T,N,3)`；旧 `energy` 是比能量，新 `total_energy` 是总能量；新结果没有 `anguluar_momentum` 键。

**实现机制：**

```text
_system_arrays 检查位置、质量、掩码
复制为当前 r、m、mask，检查并复制当前速度 v
检查 dt、steps
force = 自定义回调，或 make_nbody_acceleration(m,mask)
创建 history 字典，每个键先是空列表
record() 保存初始 r、v 的副本与诊断值
循环 steps 次：
    r,v = integrator(r,v,dt,acceleration_function=force)
    检查返回 shape 与有限值
    record() 保存本步状态
把 history 中的列表变为 NumPy 数组
加入 times 后返回 result
```

`history` 是函数局部变量，外部读取返回的 `data` / `result`，不是直接访问内部 history。`record()` 也是内部函数，不是公共 API。加速度不保存在返回字典，需要时从指定快照重新计算。

**边界与错误：**

- `steps=0` 返回一条初始记录；不会调用积分器，但初始诊断仍会计算。
- 当前 active 质量合计为零时，E/P/L 为零，质心历史是 NaN；直接调用 `center_of_mass` 则报错。
- 自定义回调覆盖默认动力学，但诊断仍按 masses/mask 计算牛顿引力总量；不自动改成自定义力的能量。
- 即使自定义力无奇点，初始 active 重合仍可能在 `total_energy` 的记录环节报错。
- 无碰撞/捕获事件或提前停止功能；全部历史存内存。手动循环可以只存选定步数。
- 不返回 names/masses/mask 元数据，调用者应另存，以免忘记行号含义。

**示例：运行、索引、计算某一时刻加速度。**

```python
import numpy as np
from orbit_sim.dynamics import M_EARTH, nbody_accelerations
from orbit_sim.integrators import Velocity_Verlet_KDK
from orbit_sim.simulate import simulate_nbody

R = np.array([[0.,0.,0.], [3.844e8,0.,0.], [1.e7,0.,0.]])
V = np.array([[0.,0.,0.], [0.,1000.,0.], [0.,6300.,0.]])
m = np.array([M_EARTH,7.342e22,1000.])
mask = np.array([True,True,False])
data = simulate_nbody(R,V,m,20.,100,Velocity_Verlet_KDK,mask)

initial_all = data["positions"][0]       # (N,3)
final_all = data["positions"][-1]       # (N,3)
moon_track = data["positions"][:,1]     # (T,3)
spacecraft_at_step_10 = data["positions"][10,2]  # (3,)
moon_x_history = data["positions"][:,1,0]       # (T,)
acceleration_at_step_10 = nbody_accelerations(data["positions"][10],m,mask)
assert moon_track.shape == (101,3)
assert acceleration_at_step_10.shape == (3,3)
np.testing.assert_array_equal(data["times"][10],200.)
np.testing.assert_array_equal(R,initial_all)  # 用户初始数组没有变。
```

**示例：自定义位置回调。** 以下自由运动只是说明覆盖默认力，不应检查其牛顿势能守恒。

```python
import numpy as np
from orbit_sim.simulate import simulate_nbody
from orbit_sim.integrators import Velocity_Verlet_KDK

R = np.array([[0.,0.,0.], [2.,0.,0.]])
V = np.array([[0.,1.,0.], [0.,1.,0.]])
def zero_acceleration(positions):
    return np.zeros_like(positions)

data = simulate_nbody(R,V,[3.,4.],1.,2,Velocity_Verlet_KDK,
                      acceleration_function=zero_acceleration)
np.testing.assert_allclose(data["positions"][-1],R+2*V)
```

<a id="diagnostics"></a>
## 7. diagnostics.py API

### 7.1 新诊断的共同选择规则

四个新公共函数都支持 `gravitating=None`，省略时默认选所有行。要与 test 模型一致，应传相同 mask，只统计相互反馈的 active 子系统。`total_energy`、P、L 的内部 `_selected` 还筛除零质量行；质心按所选质量加权，零质量贡献自然为零。

所有输入质量非负、数组有限，shape 必须对应；P 没有位置参数。**诊断不推进状态，也不缓存历史。**需要历史时由 simulate 在每一步调用。

test 有真实质量但不反馈，如果把它的动能/动量计入“封闭系统总量”，就不能期待该总量守恒。单独研究 test 轨迹可以计算它的月心相对比能量等指标，见第 11 节。

### 7.2 `[旧] specific_energy`

```text
specific_energy(position, velocity)
```

**输入/输出：** 单目标位置/速度 `(3,)`；返回标量 J/kg。

**机制：** `0.5*|velocity|² - G*M_Earth/|position|`，势能中心固定为原点 Earth。`M_Earth` 是旧 diagnostics 常量，数值与 dynamics 的 `M_EARTH` 相同。原点位置报 `ValueError`。

**适用：** 旧固定 Earth 模型。不是有限质量系统总能量，也不能直接输入航天器月心相对状态后当成 Moon 比能量（质量仍是 Earth）。

```python
import numpy as np
from orbit_sim.dynamics import G, M_EARTH
from orbit_sim.diagnostics import specific_energy

r = np.array([1.e7,0.,0.])
v = np.array([0.,np.sqrt(G*M_EARTH/1.e7),0.])
np.testing.assert_allclose(specific_energy(r,v),-G*M_EARTH/(2.e7))
```

### 7.3 `[旧] specific_angular_momentum`

```text
specific_angular_momentum(position, velocity)
```

**输入/输出：** 单目标 `(3,)`；返回 `(3,)`，m²/s。

**机制：** `np.cross(position, velocity)`，即 r × v。它本身不依赖 Earth 质量，也可对正确转换的相对 r/v 求比角动量；该量在多体相对运动中不一定守恒。旧函数没有新接口的输入校验。

```python
import numpy as np
from orbit_sim.diagnostics import specific_angular_momentum

h = specific_angular_momentum(np.array([2.,0.,0.]),np.array([0.,3.,0.]))
np.testing.assert_allclose(h,[0.,0.,6.])
```

### 7.4 `[新增] total_energy`

```text
total_energy(positions, velocities, masses, gravitating=None)
```

**参数：** 当前 R/V `(N,3)`，m `(N,)`，布尔 mask `(N,)` 可省略。

**返回：** 标量 J；不是 J/kg。

**机制：** 先选择 active 正质量行，求所有动能，再逐对计算引力势能，`j>i` 保证每对只计一次。

\[
E=\sum_{i\in active}\frac12m_i|v_i|^2
-\sum_{i<j,\ i,j\in active}\frac{Gm_im_j}{|r_i-r_j|}
\]

**错误/边界：** 所选正质量天体重合报错；没有所选正质量天体返回 0。test 不计入；外力或自定义模型下，此表达式可能不守恒。

```python
import numpy as np
from orbit_sim.dynamics import G
from orbit_sim.diagnostics import total_energy

R = np.array([[0.,0.,0.],[2.,0.,0.],[9.,9.,9.]])
V = np.array([[0.,1.,0.],[0.,-2.,0.],[1.,1.,1.]])
m = np.array([3.,4.,100.])
mask = np.array([True,True,False])
# active 动能 = 0.5*3*1² + 0.5*4*2² = 9.5；势能 = -G*3*4/2。
np.testing.assert_allclose(total_energy(R,V,m,mask),9.5-6*G)
```

### 7.5 `[新增] total_linear_momentum`

```text
total_linear_momentum(velocities, masses, gravitating=None)
```

**参数：** V `(N,3)`，m `(N,)`，mask 可省略；不需要 positions。

**返回：** 三分量总动量 `(3,)`，kg·m/s。

**机制：** 所选 active 的 `m_i*v_i` 相加。没有所选正质量行返回零向量。

```python
import numpy as np
from orbit_sim.diagnostics import total_linear_momentum

V = np.array([[0.,1.,0.],[0.,-2.,0.],[1.,1.,1.]])
P = total_linear_momentum(V,[3.,4.,100.],[True,True,False])
np.testing.assert_allclose(P,[0.,-5.,0.])
```

初始 P 常接近零，不要用 `|P-P0|/|P0|` 作为相对误差；改用非零物理尺度，例如 active 的 `sum(m_i*|v_i|)`。

### 7.6 `[新增] total_angular_momentum`

```text
total_angular_momentum(positions, velocities, masses, gravitating=None)
```

**参数：** R/V `(N,3)`，m `(N,)`，mask 可省略。

**返回：** 三分量总角动量 `(3,)`，kg·m²/s，关于当前惯性坐标原点。

**机制：** 所选 active 的 `m_i * cross(r_i,v_i)` 相加。它随参考原点改变，不是自动求“关于 Moon”的值。没有所选正质量行返回零向量。

```python
import numpy as np
from orbit_sim.diagnostics import total_angular_momentum

R = np.array([[0.,0.,0.],[2.,0.,0.],[9.,9.,9.]])
V = np.array([[0.,1.,0.],[0.,-2.,0.],[1.,1.,1.]])
L = total_angular_momentum(R,V,[3.,4.,100.],[True,True,False])
np.testing.assert_allclose(L,[0.,0.,-16.])
```

### 7.7 `[新增] center_of_mass`

```text
center_of_mass(positions, masses, gravitating=None)
```

**参数：** R `(N,3)`，m `(N,)`，mask 可省略；不需要速度。

**返回：** 质心位置 `(3,)`，m。

**机制：** 按所选质量求加权平均：

\[
R_{COM}=\frac{\sum_{active}m_i r_i}{\sum_{active}m_i}
\]

**错误：** 所选质量之和 <= 0 抛 `ValueError`。simulate_nbody 对这种情况专门记录 NaN，避免直接调用该函数。

```python
import numpy as np
from orbit_sim.diagnostics import center_of_mass

R = np.array([[0.,0.,0.],[2.,0.,0.],[9.,9.,9.]])
COM = center_of_mass(R,[3.,4.,100.],[True,True,False])
np.testing.assert_allclose(COM,[8./7.,0.,0.])
```

封闭系统的质心应匀速直线运动，只有初始总动量为零时才应近似固定：`COM(t)=COM(0)+P(0)/M*t`。

<a id="circular"></a>
## 8. circular_orbit.py：旧中心模型工具

这三个函数均未改，使用固定原点 Earth 质量。不要把它们直接当成有限质量地月圆轨道或月球中心模型工具。

| 函数签名 | 输入 | 返回 | 机制与限制 |
| --- | --- | --- | --- |
| `circular_orbit_velocity(position, velocity_direction)` | 两个 `(3,)` NumPy 向量；方向不必预先归一化 | `(3,)`，m/s | 归一化方向；检查非零半径/方向及方向垂直半径；速度大小 `sqrt(G*M_EARTH/r)` |
| `circular_orbit_period(radius)` | 正物理半径标量，m | 周期标量，s | `2*pi*sqrt(radius³/(G*M_EARTH))`；源码未添加严格半径校验，使用者给正值 |
| `analyical_circular_trajectory(radius, times)` | 半径标量与 `(T,)` 时间，s | `(positions, velocities)`，均 `(T,3)` | 固定 Earth 的 xy 平面圆轨道，t=0 位于 +x，沿 +y；保留 analyical 历史拼写 |

```python
import numpy as np
from orbit_sim.circular_orbit import (
    circular_orbit_velocity, circular_orbit_period, analyical_circular_trajectory,
)

radius = 1.e7
v = circular_orbit_velocity(np.array([radius,0.,0.]),np.array([0.,1.,0.]))
period = circular_orbit_period(radius)
R,V = analyical_circular_trajectory(radius,np.array([0.,period]))
np.testing.assert_allclose(V[0],v)
np.testing.assert_allclose(R[-1],[radius,0.,0.],atol=1.e-7)
```

<a id="internals"></a>
## 9. 内部辅助函数与调用框架

名称前的 `_` 表示内部辅助函数约定；实验脚本优先调用公共 API。以下说明帮助阅读源码，不承诺内部接口长期不变。

### 9.1 `[新增内部] _system_arrays`

```text
_system_arrays(positions, masses, gravitating=None)
    -> (positions_array, masses_array, boolean_mask)
```

**它是校验器，不是 System 类、状态仓库或加速度函数。**只接收当前位置、质量和掩码，不接收速度，不计算 a，不存时间历史。

**逐步机制：**

1. `np.asarray(..., dtype=float)` 将位置、质量整理为浮点 NumPy 数组。
2. 检查位置为二维且三列 `(N,3)`，质量为 `(N,)`。
3. 检查位置/质量有限，质量非负。
4. None 掩码生成 N 个 True；否则检查 shape 与布尔 dtype。
5. 返回三个数组。

`np.asarray` 不是“保证复制”：传入已满足类型的数组时可能共享原数据。需要独立保存的地方（simulate 初值、callback 元数据）另行 `.copy()`。因此这里的“整理”与 history 的“保存副本”是不同操作。

```python
import numpy as np
from orbit_sim.dynamics import _system_arrays  # 仅用于理解内部机制。

R,m,mask = _system_arrays([[0.,0.,0.],[2.,0.,0.]],[3.,4.])
assert R.shape == (2,3) and m.shape == (2,)
np.testing.assert_array_equal(mask,[True,True])
try:
    _system_arrays([[0.,0.,0.],[2.,0.,0.]],[3.])  # 缺少第二个质量。
except ValueError:
    pass
else:
    raise AssertionError("应拒绝位置与质量数量不一致")
```

### 9.2 `[新增内部] _prepare`

```text
_prepare(position, velocity, dt, acceleration_function)
    -> (position_array, velocity_array, force_function)
```

位置/速度转浮点数组并检查匹配 shape、有限值、有限 dt。未传回调则准备旧 Earth 函数；再包装为内部 `force(r)`：复制 r、设为只读、调用 callback、检查输出 shape/有限值。

源码变量名 `force` 虽是“力”的英文，这里的实际返回量仍是**加速度**，不是 N。它不记录轨迹，也不计算一整步。

**内部机制演示：** 返回的 force 可被调用，但实验脚本一般直接调用积分器，不需要自己调用 _prepare。

```python
import numpy as np
from orbit_sim.integrators import _prepare

R = np.array([1.,0.,0.])
V = np.zeros(3)
def callback(snapshot):
    assert not snapshot.flags.writeable  # 包装器提供只读快照。
    return -snapshot

r,v,force = _prepare(R,V,0.1,callback)
np.testing.assert_array_equal(force(r),[-1.,0.,0.])
np.testing.assert_array_equal(v,V)
```

### 9.3 `[新增内部] _selected`

```text
_selected(positions, velocities, masses, gravitating)
    -> (selected_positions, selected_velocities, selected_masses)
```

先调用 `_system_arrays`，再校验整个速度数组，最后用 `mask & (masses>0)` 选择 active 正质量行。供 E/P/L 复用。即使某行最终不选中，该行输入也必须合法、有限。P 函数因不需要真实位置，会提供同 shape 的零位置来复用此校验逻辑。

**内部机制演示：** 三行输入经过选择只剩一行；第二行虽然 mask=True，但质量为零。

```python
import numpy as np
from orbit_sim.diagnostics import _selected

R = np.array([[0.,0.,0.],[2.,0.,0.],[9.,9.,9.]])
V = np.zeros_like(R)
r,v,m = _selected(R,V,[3.,0.,100.],[True,True,False])
assert r.shape == v.shape == (1,3)
np.testing.assert_array_equal(m,[3.])
```

### 9.4 嵌套函数 acceleration / force / record

| 内部名称 | 在哪里产生 | 作用 |
| --- | --- | --- |
| `acceleration(positions)` | make_nbody_acceleration | 闭包记住质量/掩码，返回当前 a |
| `force(r)` | _prepare | 给回调只读快照，检查输出 |
| `record()` | simulate_nbody | 把当前 r/v 的 copy 与诊断值 append 到 history |

完整调用路径：

```text
用户：simulate_nbody(R,V,m,dt,steps,KDK,mask)
  ├─ _system_arrays：校验配置
  ├─ make_nbody_acceleration：产生 acceleration 回调
  ├─ record：保存初值
  └─ 每个完整时间步：
       KDK(R,V,dt,acceleration)
         ├─ _prepare：校验并包装 force
         ├─ force(R_old)
         │    acceleration(R_old)
         │      nbody_accelerations(R_old,m,mask)
         │        对每个目标调用 from_sources
         ├─ 整组更新 R_new
         ├─ force(R_new)：重新算整组加速度
         └─ 整组更新 V_new，返回
       record：保存这一步的完整状态
  最后 history 列表 → NumPy 数组 → result/data
```

系统级引力函数内部按目标循环**可以**，因为只读同一快照；按对象逐个修改位置再给下一个对象算力**不可以**，因为混用了时刻。

<a id="migration"></a>
## 10. 切换系统、参考系与旧代码迁移

### 10.1 如何换成地日系、月球系或更多天体

**不需要改 N 体函数。**改位置、速度、质量、掩码即可。函数不知道某一行叫 Sun 或 Moon。

| 模型 | 行顺序 | mask | 含义 |
| --- | --- | --- | --- |
| 地月有限质量二体 | Earth, Moon | True, True | 两者相互吸引，都推进 |
| 日地有限质量二体 | Sun, Earth | True, True | 两者相互吸引，都推进 |
| 月球 + 航天器 | Moon, spacecraft | True, False | 单一 Moon 源；Moon 无其他源施力 |
| 日地月 + 航天器 | Sun, Earth, Moon, spacecraft | True, True, True, False | 航天器受前三者；前三者不受航天器 |
| 任意有限质量系统 | 自定义 N 行 | 根据物理近似配置 | 各源相互作用，可加入多个 test |

**独立示例：日地相对圆轨道初值，Sun 初始在原点。**这是理想二体，不是真实星历。初始 Sun 速度为零，后续 Sun 会受 Earth 引力运动；整个质心允许匀速移动。

```python
import numpy as np
from orbit_sim.dynamics import G
from orbit_sim.integrators import Velocity_Verlet_KDK
from orbit_sim.simulate import simulate_nbody

m = np.array([1.9885e30,5.972e24])  # Sun, Earth
separation = 1.496e11
R = np.array([[0.,0.,0.],[separation,0.,0.]])
V = np.array([[0.,0.,0.],[0.,np.sqrt(G*m.sum()/separation),0.]])
data = simulate_nbody(R,V,m,3600.,24,Velocity_Verlet_KDK,[True,True])
assert data["positions"].shape == (25,2,3)
assert data["positions"][-1,0,0] > 0  # Sun 不是被强行固定。
```

**独立示例：固定 Moon 源 + test 航天器。**Moon 因无其他源作用且初速度为零，自然停在原点；换质量后不依赖旧 Earth 常量。

```python
import numpy as np
from orbit_sim.dynamics import G
from orbit_sim.integrators import Velocity_Verlet_KDK
from orbit_sim.simulate import simulate_nbody

m = np.array([7.342e22,1000.])
radius = 4.e6
R = np.array([[0.,0.,0.],[radius,0.,0.]])
V = np.array([[0.,0.,0.],[0.,np.sqrt(G*m[0]/radius),0.]])
data = simulate_nbody(R,V,m,10.,100,Velocity_Verlet_KDK,[True,False])
np.testing.assert_array_equal(data["positions"][:,0],np.zeros((101,3)))
```

添加 Sun 等新行时必须同时补位置、速度、质量、掩码以及自存 names。保持同一坐标系与单位；既有 closure 不会自动接受新的 N，需要重新制作。现在不支持在一次 simulate_nbody 运行中途自动增删行。

### 10.2 “中心天体”与“坐标原点”

- 主要引力源是物理配置，由质量和 mask 决定。
- 初始把某天体放原点是坐标配置，不表示它永远静止。
- 所有有限质量 active 天体默认都被传播，没有单独的 frozen-body 开关。
- `gravitating=False` 也不是 frozen-body 开关。
- 可以在惯性系推进后，用 `R_sc-R_moon` 查看月心相对轨迹。
- 不能每一步仅把 Moon 位置强行归零，仍把原有惯性加速度当作月心加速度；随动/旋转参考系的运动方程需额外考虑相对加速度或惯性项。
- 若源位置来自随时间变化的星历，需要时间依赖接口或专门传播方案；当前 position-only 回调没有自动支持该模式。

### 10.3 旧固定 Earth 调用如何迁移

**旧思路：** 只有 spacecraft 状态，Earth 不在传播数组里；积分器自己调用默认 Earth 引力。

```text
r_sc, v_sc = Velocity_Verlet_KDK(r_sc, v_sc, dt)
data = simulate_orbit(r_sc, v_sc, dt, steps, Velocity_Verlet_KDK)
```

**新思路：** 把 Earth 也放进数组，设置源身份，通过闭包或 simulate_nbody 选择新动力学。下面只含 Earth+test，故 Earth 自然固定，能够复现旧模型；再加入 active Moon 后 Earth 会运动。

```python
import numpy as np
from orbit_sim.dynamics import M_EARTH, make_nbody_acceleration
from orbit_sim.integrators import Velocity_Verlet_KDK

r_sc = np.array([1.e7,0.,0.])
v_sc = np.array([0.,1000.,0.])
old_r,old_v = Velocity_Verlet_KDK(r_sc,v_sc,10.)
R = np.vstack([np.zeros(3),r_sc])
V = np.vstack([np.zeros(3),v_sc])
a = make_nbody_acceleration([M_EARTH,1000.],[True,False])
new_R,new_V = Velocity_Verlet_KDK(R,V,10.,acceleration_function=a)
np.testing.assert_allclose(new_R[1],old_r)
np.testing.assert_allclose(new_V[1],old_v)
```

输入本来是地心/月心相对状态时，要转换两个量，不只是位置：

```text
R_sc_inertial = R_reference_inertial + relative_position
V_sc_inertial = V_reference_inertial + relative_velocity
```

读取结果时反向做差。Moon 已运动而 spacecraft 速度漏加 Moon 速度，会得到不同物理初值。

<a id="capture"></a>
## 11. 最小捕获实验模板

本模板自成一段，使用 ideal Earth/Moon 的有限质量质心初值，在 Moon 附近放 test spacecraft。可以调整 `rho0`、`u0` 扫描月心相对初值。它用于学习如何设置、传播和读取指标，**不声称这个初值就是捕获成功，更不声称永久捕获**。

```python
import numpy as np
from orbit_sim.dynamics import G, M_EARTH
from orbit_sim.integrators import Velocity_Verlet_KDK
from orbit_sim.simulate import simulate_nbody

names = ["Earth","Moon","spacecraft"]
m = np.array([M_EARTH,7.342e22,1000.])
mask = np.array([True,True,False])
d = 3.844e8
M = m[:2].sum()
speed = np.sqrt(G*M/d)
R = np.array([[-m[1]/M*d,0.,0.],[m[0]/M*d,0.,0.],[0.,0.,0.]])
V = np.array([[0.,-m[1]/M*speed,0.],[0.,m[0]/M*speed,0.],[0.,0.,0.]])

# 需要你扫描的量：月心相对位置(m)、月心相对速度(m/s)。
rho0 = np.array([-8.e6,0.,0.])
u0 = np.array([0.,1100.,0.])
R[2] = R[1] + rho0
V[2] = V[1] + u0

data = simulate_nbody(R,V,m,20.,1000,Velocity_Verlet_KDK,mask)
rho = data["positions"][:,2] - data["positions"][:,1]
u = data["velocities"][:,2] - data["velocities"][:,1]
moon_distance = np.linalg.norm(rho,axis=1)
moon_specific_energy = 0.5*np.sum(u*u,axis=1) - G*m[1]/moon_distance
moon_specific_angular_momentum = np.cross(rho,u)

# 简单的离散样本标记，不是事件检测或充分捕获判据。
inside_region = moon_distance < 2.e7  # 示例研究区域，不是自动求得的 Hill 半径。
negative_energy = moon_specific_energy < 0.
candidate_samples = inside_region & negative_energy
moon_radius = 1.7374e6
surface_intersection_samples = moon_distance <= moon_radius
print("minimum Moon distance (m):",moon_distance.min())
print("candidate samples:",np.count_nonzero(candidate_samples))
print("sampled surface intersection:",bool(np.any(surface_intersection_samples)))
assert rho.shape == u.shape == moon_specific_angular_momentum.shape == (1001,3)
```

### 11.1 哪些是现成 API，哪些是你自己写的实验指标？

- `simulate_nbody`：现成 API，负责传播与总量历史。
- `rho`、`u`：你从历史中计算的月心相对状态。
- `moon_specific_energy`：模板自行计算，使用 Moon 质量，单位 J/kg；不是调用旧 Earth `specific_energy`。
- `candidate_samples`：你定义的离散候选条件，不是基础模块提供的“捕获检测函数”。
- `surface_intersection_samples`：事后检查样本是否到达月表；不是积分中的碰撞终止功能。

负月心瞬时比能量只表示在该时刻二体近似下束缚，不保证在 Earth/Sun 摄动下持续束缚。应定义持续时间、研究区域、逃逸/撞击条件，再观察足够长的历史。不能把负能量样本数量乘以 dt 就自动当成一个连续捕获段；还需区分多个进入/离开区间并细化边界时刻。

### 11.2 需要碰撞/逃逸时提前停止怎么办？

当前 simulate_nbody 一次完成所有 steps。可自己逐整组系统调用 KDK，然后在每个完整步后检查事件。以下是独立的循环结构示例；物理半径和步长仅为演示。

```python
import numpy as np
from orbit_sim.dynamics import G,make_nbody_acceleration
from orbit_sim.integrators import Velocity_Verlet_KDK

m = np.array([7.342e22,1000.])
mask = np.array([True,False])
R = np.array([[0.,0.,0.],[4.e6,0.,0.]])
V = np.array([[0.,0.,0.],[0.,np.sqrt(G*m[0]/4.e6),0.]])
a = make_nbody_acceleration(m,mask)
dt = 10.
positions_history = [R.copy()]
times = [0.]
for step in range(100):
    R,V = Velocity_Verlet_KDK(R,V,dt,acceleration_function=a)
    positions_history.append(R.copy())
    times.append((step+1)*dt)
    if np.linalg.norm(R[1]-R[0]) <= 1.7374e6:
        print("sampled surface intersection; stop")
        break
track = np.asarray(positions_history)
assert len(track) == len(times)
```

步间可能跨过表面而两个记录点都在外面，因此仅检查离散步末不保证不漏事件；需要更小步长或设计事件定位。接近点质量中心时，数值可能先发散或报重合错误，不能期待内置碰撞处理。

建议将捕获扫描放在新的实验目录，例如 `experiments/lunar/004_capture_scan/`；让 003 始终保持基础验证用途。

<a id="validation"></a>
## 12. 时间步长、收敛与验证

### 12.1 步长选择

KDK/DKD 为光滑位置依赖模型的二阶方法；Euler 和辛 Euler 为一阶。辛结构不代表数值能量完全不变，更不代表任何 dt 都可靠。

最短近距离运动时间尺度可估算为 `sqrt(r_min³/(G*M_source))`，dt 应远小于它。地月远场可用的 dt 不一定适合航天器近月飞掠；20 s 是示例值，不是安全保证。只检查 active 守恒也不够，test 不反馈，错误的 test 轨迹不会破坏 active 守恒。

### 12.2 dt、dt/2、dt/4 到相同终止时间

在未知解析解时，可比较粗/中/细解在相同终点的差。以下示例只检查接口与对齐时间，实际二阶收敛比由 003 的长周期实验验证。

```python
import numpy as np
from orbit_sim.dynamics import M_EARTH
from orbit_sim.integrators import Velocity_Verlet_KDK
from orbit_sim.simulate import simulate_nbody

R = np.array([[0.,0.,0.],[3.844e8,0.,0.],[1.e7,0.,0.]])
V = np.array([[0.,0.,0.],[0.,1000.,0.],[0.,6300.,0.]])
m = np.array([M_EARTH,7.342e22,1000.])
mask = np.array([True,True,False])
end_time = 2000.
runs = {}
for dt in (20.,10.,5.):
    steps = round(end_time/dt)
    assert np.isclose(steps*dt,end_time)
    runs[dt] = simulate_nbody(R,V,m,dt,steps,Velocity_Verlet_KDK,mask)

coarse = runs[20.]["positions"][-1,2]
medium = runs[10.]["positions"][-1,2]
fine = runs[5.]["positions"][-1,2]
coarse_difference = np.linalg.norm(coarse-medium)
fine_difference = np.linalg.norm(medium-fine)
assert fine_difference > 0
print("spacecraft convergence ratio:",coarse_difference/fine_difference)
# 对齐整条轨迹：中解每2条、细解每4条对应粗解时刻。
np.testing.assert_allclose(runs[20.]["times"],runs[10.]["times"][::2])
np.testing.assert_allclose(runs[20.]["times"],runs[5.]["times"][::4])
```

光滑且已进入收敛区间的二阶方法，连续两组差的比通常接近 4；不是任意事件指标都必须精确为 4。近心最小距离、捕获持续时间、事件时刻还存在采样误差，应分别做细化检查。终点相同很关键，不能比较两个不同结束时间的轨迹来推断收敛。

### 12.3 守恒量的读取和误差

- 能量：`max(abs(E-E0))/abs(E0)`，前提 E0 非零。
- 角动量：`max(norm(L-L0))/norm(L0)`，前提 L0 非零。
- 动量：差的范数除以非零尺度 `sum(m_active*norm(v_active))`，不是除以近零 P0。
- 质心：与 `COM0+P0/M*times` 比较，避免错误要求它总在原点。
- 若参考能量、角动量或速度尺度为零，选明确的非零物理尺度或报告绝对误差。

```python
import numpy as np
from orbit_sim.dynamics import G,M_EARTH
from orbit_sim.integrators import Velocity_Verlet_KDK
from orbit_sim.simulate import simulate_nbody

m = np.array([M_EARTH,7.342e22])
d = 3.844e8
M = m.sum()
speed = np.sqrt(G*M/d)
R = np.array([[-m[1]/M*d,0.,0.],[m[0]/M*d,0.,0.]])
V = np.array([[0.,-m[1]/M*speed,0.],[0.,m[0]/M*speed,0.]])
data = simulate_nbody(R,V,m,100.,100,Velocity_Verlet_KDK)
E = data["total_energy"]
P = data["total_linear_momentum"]
L = data["total_angular_momentum"]
COM = data["center_of_mass"]
energy_error = np.max(np.abs(E-E[0]))/abs(E[0])
angular_error = np.max(np.linalg.norm(L-L[0],axis=1))/np.linalg.norm(L[0])
scale = np.sum(m*np.linalg.norm(V,axis=1))
momentum_error = np.max(np.linalg.norm(P-P[0],axis=1))/scale
expected_COM = COM[0] + data["times"][:,None]*P[0]/M
com_error = np.max(np.linalg.norm(COM-expected_COM,axis=1))/d
assert max(energy_error,angular_error,momentum_error,com_error) < 1.e-6
```

### 12.4 运行现有验收

本仓库当前未包含 `tests/`，表中测试项需在含测试目录的工作副本中运行。
003 验证脚本随仓库发布，路径为 `experiments/lunar/003_nbody_validation/main.py`。

在项目根目录运行：

```sh
pytest -q
python experiments/lunar/003_nbody_validation/main.py
```

若 `import orbit_sim` 失败，先按 `pyproject.toml` 安装本地包，或在项目根目录设置 `PYTHONPATH=src`。

| 验证 | 检查内容 |
| --- | --- |
| 原有测试 | 固定 Earth 引力、旧积分器、旧比能量/角动量 |
| 新低层/系统测试 | 单源与旧 Earth 一致、第三定律、test 不反馈、任意七体逐对参考值、平移/排列不变性 |
| 新积分测试 | 两种 shape、自定义力、完整 KDK/DKD 阶段快照、输入不变 |
| 新诊断/模拟测试 | 解析总量、长期守恒/质心、二阶收敛、历史 shape、零步与全 passive、非法输入、奇点 |
| 003 有限质量二体 | KDK/DKD 各 20 周期，400 步/周期；E误差 < 1e-6，L/P误差 < 1e-11 |
| 003 收敛 | 一周期 200 与 400 步的位置误差比在 3.8–4.2 |
| 003 Earth/Moon/test | 20 s × 5000 步；含 test 和不含 test 的 active 位置历史逐元素一致；航天器有限且运动 |

此前实现验收为 37 passed（原有12项 + 新增25项）。003 实际参考值：

| 方法 | 20周期最大相对E误差 | 归一化L误差 | 归一化P误差 | 位置收敛比 |
| --- | --- | --- | --- | --- |
| KDK | 1.52145e-8 | 7.91116e-15 | 4.47607e-15 | 3.9989 |
| DKD | 3.80376e-9 | 8.71842e-15 | 3.21037e-15 | 3.9981 |

这些阈值与结果属于这个理想二体实验，不是所有初值和捕获实验的通用保证。003 的 test 演示没有声称已做航天器长期收敛或捕获验证。

001/002 与 circular_orbit.py 在此前实现中通过修改前后 SHA-256 一致性检查；完整历史脚本运行通过，验证时仅在运行进程里屏蔽图形显示/保存及大量 print，未改脚本。后续请继续保留历史实验。

<a id="faq"></a>
## 13. 常见错误与英文词汇

### 13.1 使用错误速查

| 问题 | 原因与正确做法 |
| --- | --- |
| `positions` 到底是当前状态还是轨迹？ | 新 simulate 输入是 `(N,3)`；返回 `data['positions']` 才是 `(T,N,3)`；看 scope 与 shape |
| 想从外部读取 `history` | 它是局部收集容器；读取 simulate 返回值 data |
| 以为 `_system_arrays` 保存系统 | 它仅转换/校验并返回位置、质量、mask；不处理速度/加速度/历史 |
| spacecraft=False 却还在运动 | False 只禁用引力反馈；所有行都会传播 |
| Earth 初始原点，后来移动 | active Earth 会受其他 active 源作用；初始原点不是固定约束 |
| 只把状态改成二维就认为开启N体 | 积分器默认仍是固定 Earth；传通用回调或用 simulate_nbody |
| `acceleration_function=a(R)` | 错把加速度数组当函数；传 `a`，积分器自行调用 |
| 多步始终使用初始加速度 A0 | 加速度依赖所有源当前位置；每个阶段应重新求值 |
| `gravitating=[1,1,0]` 报错 | 当前实现要求布尔，写 `[True,True,False]` |
| 给源列表包含目标自身 | 低层函数不自动排自身；先排除，或用 nbody_accelerations |
| 先 update Earth 再用新 Earth 给旧 Moon 算力 | 时刻混合；整组调用积分器 |
| closure 创建后修改 masses 却无效果 | 元数据已复制；重新 make_nbody_acceleration |
| 新结果没有 `energy` / `anguluar_momentum` | 新字段是 total_energy / total_angular_momentum；单位和意义也不同 |
| 使用旧 specific_energy 求月心能量 | 它固定 Earth 质量；使用 Moon 质量和月心相对状态自行计算 |
| 把test质量计入总量后要求守恒 | 与忽略反馈的动力学不一致；诊断传相同mask |
| 仅active能量小误差就断言捕获可信 | test误差不会反馈；还要看航天器轨迹及事件指标收敛 |
| 某步出现重合异常 | 点质量奇点；检查初值、行索引、步长和事件设计，不能跳过异常继续 |
| 把dt=0传simulate | 模拟要求dt>0；单步积分器允许0，用于零步长测试 |

### 13.2 英文与 Python 阅读小词典

| 词/写法 | 含义与这里的用途 |
| --- | --- |
| body / bodies | 天体或物体 / 复数 |
| source / target | 引力源 / 受力目标 |
| active / test body | 参与引力反馈的天体 / 只受力试验体 |
| gravitating | 本接口中“是否产生引力”的开关 |
| array / shape | 数组 / 形状，例如(N,3) |
| state / snapshot | 状态 / 某一时刻的完整快照 |
| history / record | 历史 / 记录；record()把当前状态保存下来 |
| acceleration / velocity / position | 加速度 / 速度 / 位置 |
| dynamics / integrator / simulate | 动力学 / 积分器 / 模拟 |
| callback / callable | 回调 / 可调用对象；将函数交给积分器调用 |
| closure | 闭包，记住质量/掩码的函数 |
| mask | 掩码，逐项选择的布尔开关表 |
| synchronous | 同步，使用同一阶段的完整状态 |
| specific / total | 单位质量的 / 总的，例如J/kg与J |
| relative / inertial | 相对的 / 惯性的 |
| momentum / angular momentum | 动量 / 角动量 |
| center of mass / barycentric | 质心 / 以质心为参考的 |
| convergence / time step | 收敛 / 时间步长 |
| `None` | 未提供配置；各函数按自己的默认约定处理 |
| `lambda r: -r` | 简短函数；输入r，返回-r |
| `[:,1]` / `[-1,2]` | 所有时刻的第1号天体 / 最后时刻的第2号天体 |
| `copy()` / `append()` | 保存独立副本 / 在列表末尾追加 |
| `axis=1` | 沿第1号维度计算，例如(T,3)中对xyz求范数 |
| `1.e7` / `3.844e8` | 科学计数法，分别为10⁷和3.844×10⁸ |

后续维护者应先阅读本文件并核对当前源码；文档是可验证的交接依据，口头记录不替代代码与接口约定。


### 文档示例验收

本次 API 文档重写后，30 个 Python 代码块已分别在独立命名空间执行通过（包括内部机制示例）；目录锚点已核对。实验模板只验证可运行性与所列断言，不能替代针对研究目标的轨迹收敛与捕获判据验证。

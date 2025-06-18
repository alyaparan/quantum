import sys
import numpy as np
import random
import math
import time
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QComboBox, QSpinBox, QTextEdit, QTabWidget, QTableWidget,
    QTableWidgetItem, QHeaderView, QMessageBox, QLineEdit, QFileDialog,
    QSplitter, QDoubleSpinBox, QSizePolicy, QSlider, QCheckBox
)
from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QFont, QColor, QPainter, QPen, QPalette, QLinearGradient, QRadialGradient
from pyqtgraph import PlotWidget, plot, BarGraphItem
from pyqtgraph.opengl import GLViewWidget, GLLinePlotItem, GLMeshItem, GLScatterPlotItem
from pyqtgraph.opengl.MeshData import MeshData
import pyqtgraph as pg
import pyqtgraph.opengl as gl

class QuantumComputer:
    def __init__(self, num_qubits=1):
        self.num_qubits = num_qubits
        self.reset()
        self.history = []
        self.measurement_history = []
        
    def reset(self):
        """Сбрасывает компьютер в начальное состояние"""
        self.state = np.zeros(2**self.num_qubits, dtype=complex)
        self.state[0] = 1.0
        self.measured = [False] * self.num_qubits
        self.measurement_results = [[] for _ in range(self.num_qubits)]
        self.history = [('init', None)]
        self.measurement_history = []
        return f"Система сброшена в состояние |{'0'*self.num_qubits}>"
    
    def apply_gate(self, gate, target, control=None, angle=None):
        """Применяет квантовый гейт к целевым кубитам"""
        if self.measured[target]:
            return False, f"Ошибка: Кубит {target} уже измерен!"
            
        if control is not None and self.measured[control]:
            return False, f"Ошибка: Управляющий кубит {control} уже измерен!"
            
        gate = gate.upper()
        prev_state = self.state.copy()
        
        if gate == 'H':
            self._apply_hadamard(target)
        elif gate == 'X':
            self._apply_pauli_x(target)
        elif gate == 'Y':
            self._apply_pauli_y(target)
        elif gate == 'Z':
            self._apply_pauli_z(target)
        elif gate == 'S':
            self._apply_phase_gate(target, np.pi/2)
        elif gate == 'T':
            self._apply_phase_gate(target, np.pi/4)
        elif gate == 'RX' and angle is not None:
            self._apply_rx_gate(target, angle)
        elif gate == 'RY' and angle is not None:
            self._apply_ry_gate(target, angle)
        elif gate == 'RZ' and angle is not None:
            self._apply_rz_gate(target, angle)
        elif gate == 'CNOT' and control is not None:
            self._apply_cnot(control, target)
        elif gate == 'SWAP' and control is not None:
            self._apply_swap(control, target)
        elif gate == 'TOFFOLI' and control is not None and len(control) == 2:
            self._apply_toffoli(control[0], control[1], target)
        else:
            return False, f"Неизвестный гейт или параметры: {gate}"
        
        self.history.append(('gate', (gate, target, control, angle, prev_state)))
        return True, f"Применен {gate} к кубиту {target}" + (f" (управление: {control})" if control is not None else "") + (f" (угол: {angle:.2f})" if angle is not None else "")
    
    def measure(self, target, num_measurements=1):
        """Измеряет кубит"""
        if self.measured[target]:
            return None, f"Кубит {target} уже измерен!"
            
        results = []
        for _ in range(num_measurements):
            prev_state = self.state.copy()
            prob0 = self._get_probability(target, 0)
            result = 0 if random.random() < prob0 else 1
            results.append(result)
            self._collapse_state(target, result)
            self.measurement_results[target].append(result)
            self.history.append(('measure', (target, result, prev_state)))
            self.measurement_history.append((target, result, time.time()))
        
        self.measured[target] = True
        return results, f"Измерен кубит {target}: результаты = {results}"
    
    def undo(self):
        """Отменяет последнюю операцию"""
        if len(self.history) <= 1:
            return False, "Нет операций для отмены"
        
        op_type, data = self.history.pop()
        
        if op_type == 'gate' or op_type == 'measure':
            self.state = data[-1]
            if op_type == 'measure':
                target = data[0]
                self.measured[target] = False
                if self.measurement_results[target]:
                    self.measurement_results[target].pop()
                if self.measurement_history:
                    self.measurement_history.pop()
        
        return True, "Отменена последняя операция"
    
    def get_state_string(self):
        """Возвращает текущее состояние в виде строки"""
        return self._state_to_string()
    
    def get_probabilities(self):
        """Возвращает вероятности всех состояний"""
        probs = []
        for i, amp in enumerate(self.state):
            prob = abs(amp)**2
            basis = format(i, f'0{self.num_qubits}b')
            probs.append((f"|{basis}>", round(prob, 4)))
        return probs
    
    def get_qubit_states(self):
        """Возвращает состояние каждого кубита отдельно"""
        states = []
        for q in range(self.num_qubits):
            prob0 = self._get_probability(q, 0)
            states.append({
                "index": q,
                "measured": self.measured[q],
                "prob0": prob0,
                "prob1": 1 - prob0,
                "results": self.measurement_results[q]
            })
        return states
    
    def get_bloch_coordinates(self, qubit):
        """Возвращает координаты на сфере Блоха для кубита"""
        if self.measured[qubit]:
            return None, None, None
        
        dm = self._get_reduced_density_matrix(qubit)
        x = 2 * dm[0, 1].real
        y = 2 * dm[0, 1].imag
        z = dm[0, 0].real - dm[1, 1].real
        
        return x, y, z
    
    def _get_reduced_density_matrix(self, qubit):
        """Вычисляет матрицу плотности для одного кубита"""
        full_dm = np.outer(self.state, self.state.conj())
        dm = np.zeros((2, 2), dtype=complex)
        
        for i in range(2**self.num_qubits):
            for j in range(2**self.num_qubits):
                match = True
                for q in range(self.num_qubits):
                    if q != qubit:
                        bit_i = (i >> (self.num_qubits - 1 - q)) & 1
                        bit_j = (j >> (self.num_qubits - 1 - q)) & 1
                        if bit_i != bit_j:
                            match = False
                            break
                
                if match:
                    qubit_i = (i >> (self.num_qubits - 1 - qubit)) & 1
                    qubit_j = (j >> (self.num_qubits - 1 - qubit)) & 1
                    dm[qubit_i, qubit_j] += full_dm[i, j]
        
        return dm
    
    # Реализации квантовых гейтов
    def _apply_hadamard(self, target):
        H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        self._apply_single_qubit_gate(H, target)
    
    def _apply_pauli_x(self, target):
        X = np.array([[0, 1], [1, 0]])
        self._apply_single_qubit_gate(X, target)
    
    def _apply_pauli_y(self, target):
        Y = np.array([[0, -1j], [1j, 0]])
        self._apply_single_qubit_gate(Y, target)
    
    def _apply_pauli_z(self, target):
        Z = np.array([[1, 0], [0, -1]])
        self._apply_single_qubit_gate(Z, target)
    
    def _apply_phase_gate(self, target, angle):
        P = np.array([[1, 0], [0, np.exp(1j * angle)]])
        self._apply_single_qubit_gate(P, target)
    
    def _apply_rx_gate(self, target, angle):
        RX = np.array([
            [np.cos(angle/2), -1j*np.sin(angle/2)],
            [-1j*np.sin(angle/2), np.cos(angle/2)]
        ])
        self._apply_single_qubit_gate(RX, target)
    
    def _apply_ry_gate(self, target, angle):
        RY = np.array([
            [np.cos(angle/2), -np.sin(angle/2)],
            [np.sin(angle/2), np.cos(angle/2)]
        ])
        self._apply_single_qubit_gate(RY, target)
    
    def _apply_rz_gate(self, target, angle):
        RZ = np.array([[np.exp(-1j*angle/2), 0], [0, np.exp(1j*angle/2)]])
        self._apply_single_qubit_gate(RZ, target)
    
    def _apply_cnot(self, control, target):
        size = 2**self.num_qubits
        cnot_matrix = np.zeros((size, size))
        
        for i in range(size):
            control_bit = (i >> (self.num_qubits - 1 - control)) & 1
            if control_bit == 1:
                j = i ^ (1 << (self.num_qubits - 1 - target))
            else:
                j = i
            cnot_matrix[j, i] = 1
        
        self.state = np.dot(cnot_matrix, self.state)
    
    def _apply_swap(self, qubit1, qubit2):
        size = 2**self.num_qubits
        swap_matrix = np.zeros((size, size))
        
        for i in range(size):
            bit1 = (i >> (self.num_qubits - 1 - qubit1)) & 1
            bit2 = (i >> (self.num_qubits - 1 - qubit2)) & 1
            
            j = i & ~(1 << (self.num_qubits - 1 - qubit1)) & ~(1 << (self.num_qubits - 1 - qubit2))
            j |= bit1 << (self.num_qubits - 1 - qubit2)
            j |= bit2 << (self.num_qubits - 1 - qubit1)
            
            swap_matrix[j, i] = 1
        
        self.state = np.dot(swap_matrix, self.state)
    
    def _apply_toffoli(self, control1, control2, target):
        size = 2**self.num_qubits
        toffoli_matrix = np.zeros((size, size))
        
        for i in range(size):
            control_bit1 = (i >> (self.num_qubits - 1 - control1)) & 1
            control_bit2 = (i >> (self.num_qubits - 1 - control2)) & 1
            
            if control_bit1 == 1 and control_bit2 == 1:
                j = i ^ (1 << (self.num_qubits - 1 - target))
            else:
                j = i
            toffoli_matrix[j, i] = 1
        
        self.state = np.dot(toffoli_matrix, self.state)
    
    def _apply_single_qubit_gate(self, gate, target):
        full_gate = 1
        for i in range(self.num_qubits):
            if i == target:
                full_gate = np.kron(full_gate, gate)
            else:
                full_gate = np.kron(full_gate, np.eye(2))
        
        self.state = np.dot(full_gate, self.state)
    
    def _get_probability(self, target, value):
        prob = 0.0
        for i in range(len(self.state)):
            bit = (i >> (self.num_qubits - 1 - target)) & 1
            if bit == value:
                prob += abs(self.state[i])**2
        return prob
    
    def _collapse_state(self, target, result):
        for i in range(len(self.state)):
            bit = (i >> (self.num_qubits - 1 - target)) & 1
            if bit != result:
                self.state[i] = 0
        
        norm = np.sqrt(sum(abs(self.state)**2))
        if norm > 0:
            self.state /= norm
    
    def _state_to_string(self):
        state_str = []
        for i, amp in enumerate(self.state):
            if abs(amp) > 1e-6:
                basis = format(i, f'0{self.num_qubits}b')
                real_part = round(amp.real, 3)
                imag_part = round(amp.imag, 3)
                
                if abs(real_part) < 1e-3: real_part = 0
                if abs(imag_part) < 1e-3: imag_part = 0
                
                if imag_part == 0:
                    coeff = f"{real_part:.3f}"
                elif real_part == 0:
                    coeff = f"{imag_part:.3f}j"
                else:
                    coeff = f"{real_part:.3f}{imag_part:+.3f}j"
                
                state_str.append(f"{coeff}|{basis}>")
        return " + ".join(state_str) if state_str else "0"


class BlochSphere3D(gl.GLViewWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 400)
        self.setCameraPosition(distance=5)
        self.x = 0
        self.y = 0
        self.z = 0
        self.points = []
        self.vectors = []
        self.trajectory = []
        self.animation_index = 0
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.animate)
        
        # Создаем сферу Блоха
        self.create_sphere()
        
        # Оси координат
        self.add_axes()
    
    def create_sphere(self):
        """Создает 3D сферу Блоха"""
        # Сфера
        sphere = gl.GLMeshItem(
            meshdata=MeshData.sphere(rows=20, cols=20),
            color=(0.8, 0.8, 1, 0.3),
            smooth=True,
            shader='shaded'
        )
        self.addItem(sphere)
        
        # Сетка
        grid = gl.GLLinePlotItem(
            pos=np.array([
                [np.sin(theta), np.cos(theta), 0] 
                for theta in np.linspace(0, 2*np.pi, 50)
            ]),
            color=(1, 1, 1, 0.5),
            width=1
        )
        self.addItem(grid)
        
        grid2 = gl.GLLinePlotItem(
            pos=np.array([
                [0, np.sin(theta), np.cos(theta)] 
                for theta in np.linspace(0, 2*np.pi, 50)
            ]),
            color=(1, 1, 1, 0.5),
            width=1
        )
        self.addItem(grid2)
        
        grid3 = gl.GLLinePlotItem(
            pos=np.array([
                [np.sin(theta), 0, np.cos(theta)] 
                for theta in np.linspace(0, 2*np.pi, 50)
            ]),
            color=(1, 1, 1, 0.5),
            width=1
        )
        self.addItem(grid3)
    
    def add_axes(self):
        """Добавляет оси координат"""
        # Ось X (красная)
        x_axis = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0], [1.5, 0, 0]]),
            color=(1, 0, 0, 1),
            width=2
        )
        self.addItem(x_axis)
        
        # Ось Y (зеленая)
        y_axis = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0], [0, 1.5, 0]]),
            color=(0, 1, 0, 1),
            width=2
        )
        self.addItem(y_axis)
        
        # Ось Z (синяя)
        z_axis = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0], [0, 0, 1.5]]),
            color=(0, 0, 1, 1),
            width=2
        )
        self.addItem(z_axis)
        
        # Подписи осей
        x_label = gl.GLTextItem(pos=np.array([1.6, 0, 0]), text='X', color=(1, 0, 0, 1))
        y_label = gl.GLTextItem(pos=np.array([0, 1.6, 0]), text='Y', color=(0, 1, 0, 1))
        z_label = gl.GLTextItem(pos=np.array([0, 0, 1.6]), text='Z', color=(0, 0, 1, 1))
        
        self.addItem(x_label)
        self.addItem(y_label)
        self.addItem(z_label)
    
    def set_coordinates(self, x, y, z):
        """Устанавливает текущие координаты состояния"""
        self.x = x
        self.y = y
        self.z = z
        self.points.append((x, y, z))
        
        # Очищаем предыдущий вектор
        for vector in self.vectors:
            self.removeItem(vector)
        self.vectors = []
        
        # Создаем новый вектор
        vector = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0], [x, y, z]]),
            color=(1, 1, 0, 1),
            width=3
        )
        self.addItem(vector)
        self.vectors.append(vector)
        
        # Точка на конце вектора
        point = gl.GLScatterPlotItem(
            pos=np.array([[x, y, z]]),
            color=(1, 1, 0, 1),
            size=0.1
        )
        self.addItem(point)
        self.vectors.append(point)
        
        # Обновляем траекторию
        if len(self.points) > 1:
            self.trajectory.append(np.array([self.points[-2], [self.points[-1]]]))
            trajectory_line = gl.GLLinePlotItem(
                pos=np.array(self.trajectory),
                color=(1, 0.5, 0, 0.7),
                width=1
            )
            self.addItem(trajectory_line)
            self.vectors.append(trajectory_line)
    
    def clear_points(self):
        """Очищает историю точек"""
        for vector in self.vectors:
            self.removeItem(vector)
        self.vectors = []
        self.points = []
        self.trajectory = []
    
    def start_animation(self):
        """Запускает анимацию траектории"""
        if len(self.points) < 2:
            return
            
        self.animation_index = 0
        self.animation_timer.start(50)
    
    def stop_animation(self):
        """Останавливает анимацию"""
        self.animation_timer.stop()
    
    def animate(self):
        """Обновляет анимацию"""
        if self.animation_index >= len(self.points) - 1:
            self.animation_index = 0
            
        x, y, z = self.points[self.animation_index]
        self.set_coordinates(x, y, z)
        self.animation_index += 1


class QuantumComputerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Квантовый Компьютер: 3D Визуализация")
        self.setGeometry(50, 50, 1400, 900)
        
        # Инициализация квантового компьютера
        self.qc = QuantumComputer(1)
        
        # Создание центрального виджета
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Основной макет
        main_layout = QHBoxLayout(central_widget)
        
        # Левая панель - управление
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)
        
        # Правая панель - визуализация
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 5, 5, 5)
        
        main_layout.addWidget(left_panel, 30)
        main_layout.addWidget(right_panel, 70)
        
        # Настройка левой панели (управление)
        self.setup_control_panel(left_layout)
        
        # Настройка правой панели (визуализация)
        self.setup_visualization_panel(right_layout)
        
        # Статус бар
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Готов к работе. Инициализирован 1 кубит.")
        
        # Таймер для обновления визуализации
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_visualization)
        self.update_timer.start(100)  # Обновление 10 раз в секунду
        
        # Инициализация визуализации
        self.update_visualization()
    
    def setup_control_panel(self, layout):
        """Настройка панели управления"""
        # Группа инициализации
        init_group = QGroupBox("Инициализация системы")
        init_layout = QVBoxLayout(init_group)
        
        qubit_layout = QHBoxLayout()
        qubit_layout.addWidget(QLabel("Количество кубитов:"))
        self.qubit_spin = QSpinBox()
        self.qubit_spin.setRange(1, 3)  # Ограничение для производительности
        self.qubit_spin.setValue(1)
        qubit_layout.addWidget(self.qubit_spin)
        
        init_btn = QPushButton("Инициализировать")
        init_btn.clicked.connect(self.initialize_system)
        qubit_layout.addWidget(init_btn)
        
        reset_btn = QPushButton("Сбросить")
        reset_btn.clicked.connect(self.reset_system)
        qubit_layout.addWidget(reset_btn)
        
        init_layout.addLayout(qubit_layout)
        
        # Группа операций
        op_group = QGroupBox("Квантовые операции")
        op_layout = QVBoxLayout(op_group)
        
        # Выбор кубита
        qubit_layout = QHBoxLayout()
        qubit_layout.addWidget(QLabel("Целевой кубит:"))
        self.target_combo = QComboBox()
        self.target_combo.addItems([str(i) for i in range(self.qc.num_qubits)])
        qubit_layout.addWidget(self.target_combo)
        
        # Выбор гейта
        gate_layout = QHBoxLayout()
        gate_layout.addWidget(QLabel("Гейт:"))
        self.gate_combo = QComboBox()
        self.gate_combo.addItems(["H", "X", "Y", "Z", "S", "T", "RX", "RY", "RZ"])
        gate_layout.addWidget(self.gate_combo)
        
        # Параметр угла
        self.angle_spin = QDoubleSpinBox()
        self.angle_spin.setRange(0, 2 * np.pi)
        self.angle_spin.setValue(np.pi)
        self.angle_spin.setSingleStep(0.1)
        self.angle_label = QLabel("Угол (рад):")
        self.angle_label.hide()
        self.angle_spin.hide()
        
        gate_layout.addWidget(self.angle_label)
        gate_layout.addWidget(self.angle_spin)
        
        # Кнопка применения
        apply_gate_btn = QPushButton("Применить гейт")
        apply_gate_btn.clicked.connect(self.apply_gate)
        gate_layout.addWidget(apply_gate_btn)
        
        op_layout.addLayout(qubit_layout)
        op_layout.addLayout(gate_layout)
        self.gate_combo.currentTextChanged.connect(self.update_gate_ui)
        
        # Двухкубитные операции
        two_qubit_layout = QHBoxLayout()
        
        # CNOT
        cnot_layout = QVBoxLayout()
        cnot_layout.addWidget(QLabel("CNOT Gate"))
        
        cnot_inner = QHBoxLayout()
        cnot_inner.addWidget(QLabel("Управление:"))
        self.control_combo = QComboBox()
        self.control_combo.addItems([str(i) for i in range(self.qc.num_qubits)])
        cnot_inner.addWidget(self.control_combo)
        
        cnot_inner.addWidget(QLabel("Цель:"))
        self.cnot_target_combo = QComboBox()
        self.cnot_target_combo.addItems([str(i) for i in range(self.qc.num_qubits)])
        cnot_inner.addWidget(self.cnot_target_combo)
        
        apply_cnot_btn = QPushButton("Применить CNOT")
        apply_cnot_btn.clicked.connect(self.apply_cnot)
        cnot_inner.addWidget(apply_cnot_btn)
        
        cnot_layout.addLayout(cnot_inner)
        two_qubit_layout.addLayout(cnot_layout)
        
        op_layout.addLayout(two_qubit_layout)
        
        # Измерение
        measure_layout = QHBoxLayout()
        measure_layout.addWidget(QLabel("Измерить кубит:"))
        
        self.measure_combo = QComboBox()
        self.measure_combo.addItems([str(i) for i in range(self.qc.num_qubits)])
        measure_layout.addWidget(self.measure_combo)
        
        measure_layout.addWidget(QLabel("Количество:"))
        self.measure_count = QSpinBox()
        self.measure_count.setRange(1, 1000)
        self.measure_count.setValue(1)
        measure_layout.addWidget(self.measure_count)
        
        measure_btn = QPushButton("Измерить")
        measure_btn.clicked.connect(self.measure_qubit)
        measure_layout.addWidget(measure_btn)
        
        op_layout.addLayout(measure_layout)
        
        # Отмена операции
        undo_layout = QHBoxLayout()
        undo_btn = QPushButton("Отменить последнюю операцию (Undo)")
        undo_btn.clicked.connect(self.undo_operation)
        undo_layout.addWidget(undo_btn)
        
        # Группа автоматизации
        auto_group = QGroupBox("Автоматизация")
        auto_layout = QVBoxLayout(auto_group)
        
        auto_btn_layout = QHBoxLayout()
        bell_btn = QPushButton("Bell State (H + CNOT)")
        bell_btn.clicked.connect(self.apply_bell_state)
        auto_btn_layout.addWidget(bell_btn)
        
        superpos_btn = QPushButton("Суперпозиция (H Gate)")
        superpos_btn.clicked.connect(self.apply_superposition)
        auto_btn_layout.addWidget(superpos_btn)
        
        auto_layout.addLayout(auto_btn_layout)
        
        # Добавление групп в левую панель
        layout.addWidget(init_group)
        layout.addWidget(op_group)
        layout.addLayout(undo_layout)
        layout.addWidget(auto_group)
        layout.addStretch()
    
    def setup_visualization_panel(self, layout):
        """Настройка панели визуализации"""
        # Верхняя часть - 3D визуализация
        visualization_group = QGroupBox("3D Визуализация состояния")
        viz_layout = QVBoxLayout(visualization_group)
        
        # 3D сфера Блоха
        self.bloch_3d = BlochSphere3D()
        viz_layout.addWidget(self.bloch_3d)
        
        # Элементы управления 3D
        control_3d_layout = QHBoxLayout()
        
        self.bloch_qubit = QComboBox()
        self.bloch_qubit.addItems([str(i) for i in range(self.qc.num_qubits)])
        
        update_btn = QPushButton("Обновить 3D")
        update_btn.clicked.connect(self.update_bloch_3d)
        
        clear_btn = QPushButton("Очистить историю")
        clear_btn.clicked.connect(self.clear_bloch_history)
        
        animate_btn = QPushButton("Запустить анимацию")
        animate_btn.clicked.connect(self.bloch_3d.start_animation)
        
        stop_btn = QPushButton("Остановить анимацию")
        stop_btn.clicked.connect(self.bloch_3d.stop_animation)
        
        control_3d_layout.addWidget(QLabel("Кубит:"))
        control_3d_layout.addWidget(self.bloch_qubit)
        control_3d_layout.addWidget(update_btn)
        control_3d_layout.addWidget(clear_btn)
        control_3d_layout.addWidget(animate_btn)
        control_3d_layout.addWidget(stop_btn)
        
        viz_layout.addLayout(control_3d_layout)
        
        # Нижняя часть - графики
        graphs_group = QGroupBox("Графики и статистика")
        graphs_layout = QVBoxLayout(graphs_group)
        
        # Вкладки для разных графиков
        self.graph_tabs = QTabWidget()
        
        # Вкладка вероятностей
        prob_tab = QWidget()
        prob_layout = QVBoxLayout(prob_tab)
        self.prob_plot = pg.PlotWidget()
        self.prob_plot.setBackground('w')
        self.prob_plot.setTitle("Вероятности состояний", color='k')
        self.prob_plot.setLabel('left', "Вероятность")
        self.prob_plot.setLabel('bottom', "Состояния")
        prob_layout.addWidget(self.prob_plot)
        self.graph_tabs.addTab(prob_tab, "Вероятности")
        
        # Вкладка истории измерений
        history_tab = QWidget()
        history_layout = QVBoxLayout(history_tab)
        self.history_plot = pg.PlotWidget()
        self.history_plot.setBackground('w')
        self.history_plot.setTitle("История измерений", color='k')
        self.history_plot.setLabel('left', "Значение")
        self.history_plot.setLabel('bottom', "Время")
        history_layout.addWidget(self.history_plot)
        self.graph_tabs.addTab(history_tab, "История измерений")
        
        # Вкладка состояния кубитов
        qubit_tab = QWidget()
        qubit_layout = QVBoxLayout(qubit_tab)
        self.qubit_plot = pg.PlotWidget()
        self.qubit_plot.setBackground('w')
        self.qubit_plot.setTitle("Состояние кубитов", color='k')
        self.qubit_plot.setLabel('left', "Вероятность |0>")
        self.qubit_plot.setLabel('bottom', "Кубит")
        qubit_layout.addWidget(self.qubit_plot)
        self.graph_tabs.addTab(qubit_tab, "Состояние кубитов")
        
        graphs_layout.addWidget(self.graph_tabs)
        
        # Добавление в правую панель
        layout.addWidget(visualization_group, 60)
        layout.addWidget(graphs_group, 40)
    
    def update_gate_ui(self, gate):
        if gate in ['RX', 'RY', 'RZ']:
            self.angle_label.show()
            self.angle_spin.show()
        else:
            self.angle_label.hide()
            self.angle_spin.hide()
    
    def initialize_system(self):
        num_qubits = self.qubit_spin.value()
        self.qc = QuantumComputer(num_qubits)
        self.update_combos()
        self.bloch_3d.clear_points()
        self.status_bar.showMessage(f"Инициализирована система с {num_qubits} кубитами.")
        self.update_visualization()
    
    def reset_system(self):
        msg = self.qc.reset()
        self.status_bar.showMessage(msg)
        self.bloch_3d.clear_points()
        self.update_visualization()
    
    def apply_gate(self):
        target = int(self.target_combo.currentText())
        gate = self.gate_combo.currentText()
        angle = self.angle_spin.value() if gate in ['RX', 'RY', 'RZ'] else None
        
        success, message = self.qc.apply_gate(gate, target, None, angle)
        self.status_bar.showMessage(message)
        
        if success:
            self.update_visualization()
    
    def apply_cnot(self):
        control = int(self.control_combo.currentText())
        target = int(self.cnot_target_combo.currentText())
        
        if control == target:
            self.status_bar.showMessage("Ошибка: управляющий и целевой кубит не могут быть одинаковыми!")
            return
            
        success, message = self.qc.apply_gate('CNOT', target, control)
        self.status_bar.showMessage(message)
        
        if success:
            self.update_visualization()
    
    def measure_qubit(self):
        target = int(self.measure_combo.currentText())
        num_measurements = self.measure_count.value()
        
        results, message = self.qc.measure(target, num_measurements)
        self.status_bar.showMessage(message)
        
        if results is not None:
            self.update_visualization()
    
    def undo_operation(self):
        success, message = self.qc.undo()
        self.status_bar.showMessage(message)
        if success:
            self.update_visualization()
    
    def apply_bell_state(self):
        """Применяет состояние Белла (запутанность)"""
        if self.qc.num_qubits < 2:
            self.status_bar.showMessage("Ошибка: нужно минимум 2 кубита для состояния Белла")
            return
            
        self.qc.apply_gate('H', 0)
        self.qc.apply_gate('CNOT', 1, 0)
        self.status_bar.showMessage("Применено состояние Белла: H(0) + CNOT(0,1)")
        self.update_visualization()
    
    def apply_superposition(self):
        """Создает суперпозицию для выбранного кубита"""
        target = int(self.target_combo.currentText())
        self.qc.apply_gate('H', target)
        self.status_bar.showMessage(f"Применен гейт Адамара к кубиту {target}")
        self.update_visualization()
    
    def update_bloch_3d(self):
        qubit = int(self.bloch_qubit.currentText())
        x, y, z = self.qc.get_bloch_coordinates(qubit)
        if x is not None:
            self.bloch_3d.set_coordinates(x, y, z)
    
    def clear_bloch_history(self):
        self.bloch_3d.clear_points()
    
    def update_combos(self):
        n = self.qc.num_qubits
        
        # Обновляем комбобоксы
        self.target_combo.clear()
        self.target_combo.addItems([str(i) for i in range(n)])
        
        self.control_combo.clear()
        self.control_combo.addItems([str(i) for i in range(n)])
        
        self.cnot_target_combo.clear()
        self.cnot_target_combo.addItems([str(i) for i in range(n)])
        
        self.measure_combo.clear()
        self.measure_combo.addItems([str(i) for i in range(n)])
        
        self.bloch_qubit.clear()
        self.bloch_qubit.addItems([str(i) for i in range(n)])
    
    def update_visualization(self):
        """Обновляет все элементы визуализации"""
        # Обновляем 3D сферу Блоха
        self.update_bloch_3d()
        
        # Обновляем график вероятностей
        self.update_probability_plot()
        
        # Обновляем график истории измерений
        self.update_measurement_history_plot()
        
        # Обновляем график состояния кубитов
        self.update_qubit_state_plot()
    
    def update_probability_plot(self):
        """Обновляет график вероятностей состояний"""
        self.prob_plot.clear()
        
        probs = self.qc.get_probabilities()
        if not probs:
            return
            
        states = [state for state, _ in probs]
        probabilities = [prob * 100 for _, prob in probs]
        
        # Создаем цветовую схему
        colors = []
        for i in range(len(states)):
            r = i / len(states)
            g = 1 - r
            b = 0.5
            colors.append((r, g, b, 0.8))
        
        # Создаем график
        bg = BarGraphItem(
            x=range(len(states)), 
            height=probabilities, 
            width=0.6, 
            brushes=colors
        )
        
        self.prob_plot.addItem(bg)
        self.prob_plot.getAxis('bottom').setTicks([[(i, state) for i, state in enumerate(states)]])
        self.prob_plot.setYRange(0, 100)
        self.prob_plot.setTitle(f"Вероятности состояний (всего {len(states)} состояний)")
    
    def update_measurement_history_plot(self):
        """Обновляет график истории измерений"""
        self.history_plot.clear()
        
        if not self.qc.measurement_history:
            return
            
        # Группируем измерения по времени
        history_by_time = {}
        for target, result, timestamp in self.qc.measurement_history:
            if timestamp not in history_by_time:
                history_by_time[timestamp] = [0, 0]
            history_by_time[timestamp][result] += 1
        
        # Сортируем по времени
        sorted_times = sorted(history_by_time.keys())
        times = []
        values_0 = []
        values_1 = []
        
        for t in sorted_times:
            times.append(t)
            values_0.append(history_by_time[t][0])
            values_1.append(history_by_time[t][1])
        
        # Создаем графики
        self.history_plot.plot(times, values_0, pen='b', symbol='o', symbolPen='b', symbolBrush='b', name='0')
        self.history_plot.plot(times, values_1, pen='r', symbol='o', symbolPen='r', symbolBrush='r', name='1')
        
        # Настройки графика
        self.history_plot.addLegend()
        self.history_plot.setTitle("История измерений")
        self.history_plot.setLabel('left', "Количество")
        self.history_plot.setLabel('bottom', "Время")
    
    def update_qubit_state_plot(self):
        """Обновляет график состояния кубитов"""
        self.qubit_plot.clear()
        
        qubit_states = self.qc.get_qubit_states()
        if not qubit_states:
            return
            
        qubits = [state['index'] for state in qubit_states]
        prob0 = [state['prob0'] * 100 for state in qubit_states]
        prob1 = [state['prob1'] * 100 for state in qubit_states]
        
        # Создаем графики
        bg0 = BarGraphItem(
            x=qubits, 
            height=prob0, 
            width=0.4, 
            brushes=(0, 0, 1, 0.7)
        )
        
        bg1 = BarGraphItem(
            x=[q + 0.4 for q in qubits], 
            height=prob1, 
            width=0.4, 
            brushes=(1, 0, 0, 0.7)
        )
        
        self.qubit_plot.addItem(bg0)
        self.qubit_plot.addItem(bg1)
        
        # Настройки графика
        self.qubit_plot.setXRange(-0.5, len(qubits)-0.5)
        self.qubit_plot.setYRange(0, 100)
        self.qubit_plot.setTitle("Состояние кубитов")
        self.qubit_plot.setLabel('left', "Вероятность (%)")
        self.qubit_plot.setLabel('bottom', "Кубит")
        self.qubit_plot.getAxis('bottom').setTicks([[(i, str(i)) for i in qubits]])


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Настройка темной темы
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.WindowText, Qt.white)
    dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
    dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
    dark_palette.setColor(QPalette.ToolTipText, Qt.white)
    dark_palette.setColor(QPalette.Text, Qt.white)
    dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ButtonText, Qt.white)
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(dark_palette)
    
    # Настройка шрифта
    font = QFont()
    font.setPointSize(10)
    app.setFont(font)
    
    window = QuantumComputerGUI()
    window.show()
    sys.exit(app.exec_())

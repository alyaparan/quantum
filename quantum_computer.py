import sys
import numpy as np
import random
import json
import math
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QComboBox, QSpinBox, QTextEdit, QTabWidget, QTableWidget,
    QTableWidgetItem, QHeaderView, QMessageBox, QLineEdit, QFileDialog,
    QSplitter, QDoubleSpinBox, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QColor, QPainter, QPen

class QuantumComputer:
    def __init__(self, num_qubits=1):
        self.num_qubits = num_qubits
        self.reset()
        self.history = []
        self.algorithm_steps = []
        
    def reset(self):
        """Сбрасывает компьютер в начальное состояние"""
        self.state = np.zeros(2**self.num_qubits, dtype=complex)
        self.state[0] = 1.0
        self.measured = [False] * self.num_qubits
        self.measurement_results = [[] for _ in range(self.num_qubits)]
        self.history = [('init', None)]
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
        
        return True, "Отменена последняя операция"
    
    def add_algorithm_step(self, step):
        """Добавляет шаг алгоритма"""
        self.algorithm_steps.append(step)
        return f"Добавлен шаг алгоритма: {step}"
    
    def run_algorithm(self):
        """Выполняет сохраненный алгоритм"""
        if not self.algorithm_steps:
            return False, "Алгоритм не задан"
        
        results = []
        for step in self.algorithm_steps:
            if step['type'] == 'gate':
                self.apply_gate(step['gate'], step['target'], step.get('control'), step.get('angle'))
            elif step['type'] == 'measure':
                result, _ = self.measure(step['target'], step.get('num_measurements', 1))
                results.extend(result)
        
        return True, f"Алгоритм выполнен. Результаты: {results}"
    
    def save_state(self, filename):
        """Сохраняет текущее состояние в файл"""
        data = {
            'num_qubits': self.num_qubits,
            'state': [complex(x) for x in self.state],
            'measured': self.measured,
            'measurement_results': self.measurement_results
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f)
        
        return f"Состояние сохранено в {filename}"
    
    def load_state(self, filename):
        """Загружает состояние из файла"""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self.num_qubits = data['num_qubits']
        self.state = np.array([complex(x) for x in data['state']])
        self.measured = data['measured']
        self.measurement_results = data['measurement_results']
        
        return f"Состояние загружено из {filename}"
    
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


class BlochSphereWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 300)
        self.x = 0
        self.y = 0
        self.z = 0
        self.points = []
        
    def set_coordinates(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.points.append((x, y, z))
        self.update()
        
    def clear_points(self):
        self.points = []
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        width = self.width()
        height = self.height()
        radius = min(width, height) * 0.4
        center_x = width / 2
        center_y = height / 2
        
        # Сфера
        painter.setPen(QPen(Qt.lightGray, 1, Qt.DashLine))
        painter.drawEllipse(int(center_x - radius), int(center_y - radius), 
                          int(radius * 2), int(radius * 2))
        
        # Оси
        painter.setPen(QPen(Qt.red, 2))
        painter.drawLine(int(center_x - radius), int(center_y), 
                       int(center_x + radius), int(center_y))
        painter.setPen(QPen(Qt.green, 2))
        painter.drawLine(int(center_x), int(center_y - radius), 
                       int(center_x), int(center_y + radius))
        painter.setPen(QPen(Qt.blue, 2))
        painter.drawLine(int(center_x), int(center_y), 
                       int(center_x), int(center_y))
        
        # Подписи осей
        font = painter.font()
        font.setPointSize(10)
        painter.setFont(font)
        painter.setPen(QPen(Qt.red, 2))
        painter.drawText(int(center_x + radius + 5), int(center_y), "X")
        painter.setPen(QPen(Qt.green, 2))
        painter.drawText(int(center_x), int(center_y - radius - 5), "Y")
        painter.setPen(QPen(Qt.blue, 2))
        painter.drawText(int(center_x), int(center_y + radius + 15), "Z")
        
        # Точки
        for i, (x, y, z) in enumerate(self.points):
            screen_x = center_x + x * radius
            screen_y = center_y - y * radius
            color = QColor(0, 0, int(255 * (z + 1)/2))
            painter.setPen(QPen(color, 3))
            size = 6 - (len(self.points) - i - 1) * 0.5
            if size < 3: size = 3
            painter.drawEllipse(int(screen_x - size/2), int(screen_y - size/2), 
                              int(size), int(size))
        
        # Текущая точка
        if self.x is not None:
            screen_x = center_x + self.x * radius
            screen_y = center_y - self.y * radius
            
            painter.setPen(QPen(Qt.black, 2))
            painter.drawLine(int(center_x), int(center_y), int(screen_x), int(screen_y))
            
            painter.setPen(QPen(Qt.black, 3))
            painter.setBrush(Qt.yellow)
            painter.drawEllipse(int(screen_x - 5), int(screen_y - 5), 10, 10)
            
            state = f"|ψ> = {1+self.z:.2f}|0> + {1-self.z:.2f}|1>"
            painter.drawText(int(screen_x + 10), int(screen_y), state)


class QuantumComputerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Квантовый Компьютер")
        self.setGeometry(100, 100, 1000, 700)
        
        self.qc = QuantumComputer(1)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        self.setup_control_tab()
        self.setup_state_tab()
        self.setup_algorithm_tab()
        self.setup_bloch_tab()
        self.setup_info_tab()
        
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Готов к работе. Инициализирован 1 кубит.")
        self.update_display()
    
    def setup_control_tab(self):
        control_tab = QWidget()
        self.tab_widget.addTab(control_tab, "Управление")
        layout = QVBoxLayout(control_tab)
        
        # Инициализация
        init_group = QGroupBox("Инициализация системы")
        init_layout = QHBoxLayout(init_group)
        
        self.qubit_spin = QSpinBox()
        self.qubit_spin.setRange(1, 5)
        self.qubit_spin.setValue(1)
        
        init_btn = QPushButton("Инициализировать")
        init_btn.clicked.connect(self.initialize_system)
        
        reset_btn = QPushButton("Сбросить")
        reset_btn.clicked.connect(self.reset_system)
        
        undo_btn = QPushButton("Отменить")
        undo_btn.clicked.connect(self.undo_operation)
        
        save_btn = QPushButton("Сохранить состояние")
        save_btn.clicked.connect(self.save_state)
        
        load_btn = QPushButton("Загрузить состояние")
        load_btn.clicked.connect(self.load_state)
        
        init_layout.addWidget(QLabel("Количество кубитов:"))
        init_layout.addWidget(self.qubit_spin)
        init_layout.addWidget(init_btn)
        init_layout.addWidget(reset_btn)
        init_layout.addWidget(undo_btn)
        init_layout.addWidget(save_btn)
        init_layout.addWidget(load_btn)
        
        # Операции
        op_group = QGroupBox("Квантовые операции")
        op_layout = QVBoxLayout(op_group)
        
        # Однокубитные операции
        single_qubit_layout = QHBoxLayout()
        single_qubit_layout.addWidget(QLabel("Целевой кубит:"))
        self.target_combo = QComboBox()
        self.target_combo.addItems([str(i) for i in range(self.qc.num_qubits)])
        single_qubit_layout.addWidget(self.target_combo)
        
        single_qubit_layout.addWidget(QLabel("Гейт:"))
        self.gate_combo = QComboBox()
        self.gate_combo.addItems(["H", "X", "Y", "Z", "S", "T", "RX", "RY", "RZ"])
        single_qubit_layout.addWidget(self.gate_combo)
        
        self.angle_spin = QDoubleSpinBox()
        self.angle_spin.setRange(0, 2 * np.pi)
        self.angle_spin.setValue(np.pi)
        self.angle_spin.setSingleStep(0.1)
        self.angle_label = QLabel("Угол (рад):")
        self.angle_label.hide()
        self.angle_spin.hide()
        
        single_qubit_layout.addWidget(self.angle_label)
        single_qubit_layout.addWidget(self.angle_spin)
        
        apply_gate_btn = QPushButton("Применить гейт")
        apply_gate_btn.clicked.connect(self.apply_gate)
        single_qubit_layout.addWidget(apply_gate_btn)
        
        op_layout.addLayout(single_qubit_layout)
        self.gate_combo.currentTextChanged.connect(self.update_gate_ui)
        
        # Многокубитные операции
        multi_qubit_layout = QHBoxLayout()
        
        # CNOT
        cnot_layout = QVBoxLayout()
        cnot_layout.addWidget(QLabel("CNOT:"))
        
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
        multi_qubit_layout.addLayout(cnot_layout)
        
        # SWAP
        swap_layout = QVBoxLayout()
        swap_layout.addWidget(QLabel("SWAP:"))
        
        swap_inner = QHBoxLayout()
        swap_inner.addWidget(QLabel("Кубит 1:"))
        self.swap1_combo = QComboBox()
        self.swap1_combo.addItems([str(i) for i in range(self.qc.num_qubits)])
        swap_inner.addWidget(self.swap1_combo)
        
        swap_inner.addWidget(QLabel("Кубит 2:"))
        self.swap2_combo = QComboBox()
        self.swap2_combo.addItems([str(i) for i in range(self.qc.num_qubits)])
        swap_inner.addWidget(self.swap2_combo)
        
        apply_swap_btn = QPushButton("Применить SWAP")
        apply_swap_btn.clicked.connect(self.apply_swap)
        swap_inner.addWidget(apply_swap_btn)
        
        swap_layout.addLayout(swap_inner)
        multi_qubit_layout.addLayout(swap_layout)
        
        op_layout.addLayout(multi_qubit_layout)
        
        # Измерение
        measure_layout = QHBoxLayout()
        measure_layout.addWidget(QLabel("Измерить кубит:"))
        
        self.measure_combo = QComboBox()
        self.measure_combo.addItems([str(i) for i in range(self.qc.num_qubits)])
        measure_layout.addWidget(self.measure_combo)
        
        measure_layout.addWidget(QLabel("Количество измерений:"))
        self.measure_count = QSpinBox()
        self.measure_count.setRange(1, 1000)
        self.measure_count.setValue(1)
        measure_layout.addWidget(self.measure_count)
        
        measure_btn = QPushButton("Измерить")
        measure_btn.clicked.connect(self.measure_qubit)
        measure_layout.addWidget(measure_btn)
        
        op_layout.addLayout(measure_layout)
        
        layout.addWidget(init_group)
        layout.addWidget(op_group)
        layout.addStretch()
    
    def setup_state_tab(self):
        state_tab = QWidget()
        self.tab_widget.addTab(state_tab, "Состояние")
        layout = QVBoxLayout(state_tab)
        
        splitter = QSplitter(Qt.Vertical)
        
        # Вектор состояния
        state_group = QGroupBox("Вектор состояния")
        state_layout = QVBoxLayout(state_group)
        self.state_display = QTextEdit()
        self.state_display.setReadOnly(True)
        self.state_display.setFont(QFont("Courier New", 10))
        state_layout.addWidget(self.state_display)
        
        # Вероятности
        prob_group = QGroupBox("Вероятности состояний")
        prob_layout = QVBoxLayout(prob_group)
        self.prob_table = QTableWidget()
        self.prob_table.setColumnCount(2)
        self.prob_table.setHorizontalHeaderLabels(["Состояние", "Вероятность"])
        self.prob_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.prob_table.verticalHeader().setVisible(False)
        prob_layout.addWidget(self.prob_table)
        
        # Состояния кубитов
        qubit_group = QGroupBox("Состояния кубитов")
        qubit_layout = QVBoxLayout(qubit_group)
        self.qubit_table = QTableWidget()
        self.qubit_table.setColumnCount(5)
        self.qubit_table.setHorizontalHeaderLabels(["Кубит", "Измерен", "P(|0>)", "P(|1>)", "Результаты"])
        self.qubit_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.qubit_table.verticalHeader().setVisible(False)
        qubit_layout.addWidget(self.qubit_table)
        
        splitter.addWidget(state_group)
        splitter.addWidget(prob_group)
        splitter.addWidget(qubit_group)
        splitter.setSizes([300, 300, 200])
        
        layout.addWidget(splitter)
    
    def setup_algorithm_tab(self):
        algorithm_tab = QWidget()
        self.tab_widget.addTab(algorithm_tab, "Алгоритмы")
        layout = QVBoxLayout(algorithm_tab)
        
        # Управление алгоритмами
        control_group = QGroupBox("Управление алгоритмами")
        control_layout = QHBoxLayout(control_group)
        
        self.algorithm_name = QLineEdit("Мой алгоритм")
        self.algorithm_name.setPlaceholderText("Название алгоритма")
        
        add_step_btn = QPushButton("Добавить шаг")
        add_step_btn.clicked.connect(self.add_algorithm_step)
        
        run_btn = QPushButton("Выполнить алгоритм")
        run_btn.clicked.connect(self.run_algorithm)
        
        clear_btn = QPushButton("Очистить алгоритм")
        clear_btn.clicked.connect(self.clear_algorithm)
        
        control_layout.addWidget(self.algorithm_name)
        control_layout.addWidget(add_step_btn)
        control_layout.addWidget(run_btn)
        control_layout.addWidget(clear_btn)
        
        # Шаги алгоритма
        steps_group = QGroupBox("Шаги алгоритма")
        steps_layout = QVBoxLayout(steps_group)
        self.algorithm_table = QTableWidget()
        self.algorithm_table.setColumnCount(5)
        self.algorithm_table.setHorizontalHeaderLabels(["Тип", "Гейт", "Цель", "Управление", "Параметры"])
        self.algorithm_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.algorithm_table.verticalHeader().setVisible(False)
        steps_layout.addWidget(self.algorithm_table)
        
        layout.addWidget(control_group)
        layout.addWidget(steps_group)
    
    def setup_bloch_tab(self):
        bloch_tab = QWidget()
        self.tab_widget.addTab(bloch_tab, "Сфера Блоха")
        layout = QVBoxLayout(bloch_tab)
        
        # Управление
        control_group = QGroupBox("Управление визуализацией")
        control_layout = QHBoxLayout(control_group)
        
        self.bloch_qubit = QComboBox()
        self.bloch_qubit.addItems([str(i) for i in range(self.qc.num_qubits)])
        
        update_btn = QPushButton("Обновить")
        update_btn.clicked.connect(self.update_bloch_sphere)
        
        clear_btn = QPushButton("Очистить историю")
        clear_btn.clicked.connect(self.clear_bloch_history)
        
        control_layout.addWidget(QLabel("Кубит:"))
        control_layout.addWidget(self.bloch_qubit)
        control_layout.addWidget(update_btn)
        control_layout.addWidget(clear_btn)
        
        # Сфера Блоха
        self.bloch_widget = BlochSphereWidget()
        
        layout.addWidget(control_group)
        layout.addWidget(self.bloch_widget)
    
    def setup_info_tab(self):
        info_tab = QWidget()
        self.tab_widget.addTab(info_tab, "Справка")
        layout = QVBoxLayout(info_tab)
        
        info_text = QTextEdit()
        info_text.setReadOnly(True)
        info_text.setHtml("""
        <h2>Квантовый Компьютер - Справка</h2>
        
        <h3>Основные понятия:</h3>
        <p><b>Кубит</b> - квантовый бит, который может находиться в суперпозиции состояний |0> и |1>.</p>
        <p><b>Суперпозиция</b> - одновременное нахождение кубита в нескольких состояниях.</p>
        <p><b>Запутанность</b> - квантовая связь между кубитами, когда состояние одного зависит от состояния другого.</p>
        
        <h3>Квантовые гейты:</h3>
        <ul>
            <li><b>H (Адамара)</b>: Создает суперпозицию: |0> → (|0> + |1>)/√2, |1> → (|0> - |1>)/√2</li>
            <li><b>X (NOT)</b>: Инвертирует состояние: |0> → |1>, |1> → |0></li>
            <li><b>Y</b>: Поворот вокруг оси Y: |0> → i|1>, |1> → -i|0></li>
            <li><b>Z</b>: Изменяет фазу: |0> → |0>, |1> → -|1></li>
            <li><b>S</b>: Фазовый гейт (π/2): |0> → |0>, |1> → i|1></li>
            <li><b>T</b>: Гейт π/8 (π/4): |0> → |0>, |1> → e^(iπ/4)|1></li>
            <li><b>RX(θ)</b>: Вращение вокруг оси X на угол θ</li>
            <li><b>RY(θ)</b>: Вращение вокруг оси Y на угол θ</li>
            <li><b>RZ(θ)</b>: Вращение вокруг оси Z на угол θ</li>
            <li><b>CNOT</b>: Управляемый NOT. Если управляющий кубит |1>, инвертирует целевой кубит.</li>
            <li><b>SWAP</b>: Обменивает состояния двух кубитов</li>
        </ul>
        
        <h3>Примеры алгоритмов:</h3>
        <p><b>Запутанность (Bell State):</b></p>
        <ol>
            <li>H на кубит 0</li>
            <li>CNOT (0 → 1)</li>
        </ol>
        
        <p><b>Квантовое случайное число:</b></p>
        <ol>
            <li>H на кубит 0</li>
            <li>Измерить кубит 0</li>
        </ol>
        """)
        
        layout.addWidget(info_text)
    
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
        self.bloch_widget.clear_points()
        self.status_bar.showMessage(f"Инициализирована система с {num_qubits} кубитами.")
        self.update_display()
    
    def reset_system(self):
        msg = self.qc.reset()
        self.status_bar.showMessage(msg)
        self.bloch_widget.clear_points()
        self.update_display()
    
    def apply_gate(self):
        target = int(self.target_combo.currentText())
        gate = self.gate_combo.currentText()
        angle = self.angle_spin.value() if gate in ['RX', 'RY', 'RZ'] else None
        
        success, message = self.qc.apply_gate(gate, target, None, angle)
        self.status_bar.showMessage(message)
        
        if success:
            self.update_display()
            self.update_bloch_sphere()
    
    def apply_cnot(self):
        control = int(self.control_combo.currentText())
        target = int(self.cnot_target_combo.currentText())
        
        if control == target:
            self.status_bar.showMessage("Ошибка: управляющий и целевой кубит не могут быть одинаковыми!")
            return
            
        success, message = self.qc.apply_gate('CNOT', target, control)
        self.status_bar.showMessage(message)
        
        if success:
            self.update_display()
            self.update_bloch_sphere()
    
    def apply_swap(self):
        qubit1 = int(self.swap1_combo.currentText())
        qubit2 = int(self.swap2_combo.currentText())
        
        if qubit1 == qubit2:
            self.status_bar.showMessage("Ошибка: кубиты для обмена должны быть разными!")
            return
            
        success, message = self.qc.apply_gate('SWAP', qubit1, qubit2)
        self.status_bar.showMessage(message)
        
        if success:
            self.update_display()
            self.update_bloch_sphere()
    
    def measure_qubit(self):
        target = int(self.measure_combo.currentText())
        num_measurements = self.measure_count.value()
        
        results, message = self.qc.measure(target, num_measurements)
        self.status_bar.showMessage(message)
        
        if results is not None:
            self.update_display()
            self.update_bloch_sphere()
    
    def undo_operation(self):
        success, message = self.qc.undo()
        self.status_bar.showMessage(message)
        if success:
            self.update_display()
            self.update_bloch_sphere()
    
    def save_state(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Сохранить состояние", "", "Quantum State Files (*.qstate)")
        if filename:
            if not filename.endswith('.qstate'):
                filename += '.qstate'
            message = self.qc.save_state(filename)
            self.status_bar.showMessage(message)
    
    def load_state(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Загрузить состояние", "", "Quantum State Files (*.qstate)")
        if filename:
            message = self.qc.load_state(filename)
            self.update_combos()
            self.update_display()
            self.update_bloch_sphere()
            self.status_bar.showMessage(message)
    
    def add_algorithm_step(self):
        # Простой способ добавления шага - в реальном приложении можно сделать диалог
        step = {
            'type': 'gate',
            'gate': 'H',
            'target': 0
        }
        self.qc.add_algorithm_step(step)
        self.update_algorithm_table()
        self.status_bar.showMessage("Шаг алгоритма добавлен")
    
    def run_algorithm(self):
        success, message = self.qc.run_algorithm()
        self.status_bar.showMessage(message)
        if success:
            self.update_display()
            self.update_bloch_sphere()
    
    def clear_algorithm(self):
        self.qc.algorithm_steps = []
        self.update_algorithm_table()
        self.status_bar.showMessage("Алгоритм очищен")
    
    def update_algorithm_table(self):
        self.algorithm_table.setRowCount(len(self.qc.algorithm_steps))
        for i, step in enumerate(self.qc.algorithm_steps):
            self.algorithm_table.setItem(i, 0, QTableWidgetItem(step['type']))
            self.algorithm_table.setItem(i, 1, QTableWidgetItem(step.get('gate', '')))
            self.algorithm_table.setItem(i, 2, QTableWidgetItem(str(step.get('target', ''))))
            self.algorithm_table.setItem(i, 3, QTableWidgetItem(str(step.get('control', ''))))
            self.algorithm_table.setItem(i, 4, QTableWidgetItem(str(step.get('angle', ''))))
    
    def update_bloch_sphere(self):
        qubit = int(self.bloch_qubit.currentText())
        x, y, z = self.qc.get_bloch_coordinates(qubit)
        if x is not None:
            self.bloch_widget.set_coordinates(x, y, z)
    
    def clear_bloch_history(self):
        self.bloch_widget.clear_points()
    
    def update_combos(self):
        n = self.qc.num_qubits
        
        self.target_combo.clear()
        self.target_combo.addItems([str(i) for i in range(n)])
        
        self.control_combo.clear()
        self.control_combo.addItems([str(i) for i in range(n)])
        
        self.cnot_target_combo.clear()
        self.cnot_target_combo.addItems([str(i) for i in range(n)])
        
        self.measure_combo.clear()
        self.measure_combo.addItems([str(i) for i in range(n)])
        
        self.swap1_combo.clear()
        self.swap1_combo.addItems([str(i) for i in range(n)])
        self.swap2_combo.clear()
        self.swap2_combo.addItems([str(i) for i in range(n)])
        
        self.bloch_qubit.clear()
        self.bloch_qubit.addItems([str(i) for i in range(n)])
    
    def update_display(self):
        # Вектор состояния
        state_str = self.qc.get_state_string()
        self.state_display.setText(state_str)
        
        # Вероятности
        probs = self.qc.get_probabilities()
        self.prob_table.setRowCount(len(probs))
        
        for i, (state, prob) in enumerate(probs):
            state_item = QTableWidgetItem(state)
            state_item.setTextAlignment(Qt.AlignCenter)
            self.prob_table.setItem(i, 0, state_item)
            
            prob_item = QTableWidgetItem(f"{prob*100:.2f}%")
            prob_item.setTextAlignment(Qt.AlignCenter)
            
            if prob > 0.5:
                prob_item.setBackground(QColor(173, 255, 173))
            elif prob > 0.1:
                prob_item.setBackground(QColor(255, 255, 173))
            
            self.prob_table.setItem(i, 1, prob_item)
        
        # Состояния кубитов
        qubit_states = self.qc.get_qubit_states()
        self.qubit_table.setRowCount(len(qubit_states))
        
        for i, state in enumerate(qubit_states):
            index_item = QTableWidgetItem(str(state["index"]))
            index_item.setTextAlignment(Qt.AlignCenter)
            self.qubit_table.setItem(i, 0, index_item)
            
            measured_item = QTableWidgetItem("Да" if state["measured"] else "Нет")
            measured_item.setTextAlignment(Qt.AlignCenter)
            if state["measured"]:
                measured_item.setBackground(QColor(255, 200, 200))
            self.qubit_table.setItem(i, 1, measured_item)
            
            prob0_item = QTableWidgetItem(f"{state['prob0']*100:.2f}%")
            prob0_item.setTextAlignment(Qt.AlignCenter)
            self.qubit_table.setItem(i, 2, prob0_item)
            
            prob1_item = QTableWidgetItem(f"{state['prob1']*100:.2f}%")
            prob1_item.setTextAlignment(Qt.AlignCenter)
            self.qubit_table.setItem(i, 3, prob1_item)
            
            results = ", ".join(map(str, state["results"])) if state["results"] else "-"
            results_item = QTableWidgetItem(results)
            results_item.setTextAlignment(Qt.AlignCenter)
            self.qubit_table.setItem(i, 4, results_item)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    font = QFont()
    font.setPointSize(10)
    app.setFont(font)
    
    window = QuantumComputerGUI()
    window.show()
    sys.exit(app.exec_())

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backend_bases import MouseEvent
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import serial
from serial.tools import list_ports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque
import time
import pandas as pd
import queue
import threading
import struct
import neurokit2 as nk
from scipy.signal import find_peaks


class Oscilloscope:
    def __init__(self, master):
        self.after_id = None
        self.master = master
        self.ser = None
        self.is_running = False
        self.paused = False
        self.buffer_size = 5000
        self.total_points = 0
        self.num_channels = 4
        self.channels = []
        for i in range(self.num_channels):
            self.channels.append({
                'color': ['blue', 'green', 'red', 'purple'][i],
                'raw_data': deque(maxlen=self.buffer_size),
                'filtered_data': deque(maxlen=self.buffer_size),
                'lpf_states': None,
                'hpf_states': None,
                'bpf_sections': None
            })
        self.timestamps = deque(maxlen=self.buffer_size)
        self.start_time = None
        self.sample_rate = 853.0
        self.sample_period = 1.0 / self.sample_rate
        self.data_queue = queue.Queue()
        self.filter_params = {
            'type': 'None',
            'cutoff': 50.0
        }
        self.buffer = bytearray()

        self.setup_gui()
        self.setup_plots()
        self.refresh_ports()

        self.master.bind('+', self.increase_scale)
        self.master.bind('-', self.decrease_scale)

        master.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self):
        self.stop()
        self.master.destroy()

    def setup_gui(self):
        main_panel = ttk.PanedWindow(self.master, orient=tk.HORIZONTAL)
        main_panel.pack(fill=tk.BOTH, expand=True)

        # Левая панель с информацией о пациенте
        left_info_panel = ttk.Frame(main_panel, width=250)
        main_panel.add(left_info_panel)

        # Правая панель с графиками и управлением
        right_panel = ttk.Frame(main_panel)
        main_panel.add(right_panel)

        # Информация о пациенте
        patient_frame = ttk.LabelFrame(left_info_panel, text="Информация о пациенте")
        patient_frame.pack(padx=10, pady=10, fill=tk.X)

        ttk.Label(patient_frame, text="Фамилия:").grid(row=0, column=0, sticky=tk.W)
        self.surname_entry = ttk.Entry(patient_frame)
        self.surname_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=2)

        ttk.Label(patient_frame, text="Имя:").grid(row=1, column=0, sticky=tk.W)
        self.name_entry = ttk.Entry(patient_frame)
        self.name_entry.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=2)

        ttk.Label(patient_frame, text="Отчество:").grid(row=2, column=0, sticky=tk.W)
        self.patronymic_entry = ttk.Entry(patient_frame)
        self.patronymic_entry.grid(row=2, column=1, sticky=tk.EW, padx=5, pady=2)

        ttk.Label(patient_frame, text="Пол:").grid(row=3, column=0, sticky=tk.W)
        self.gender_combo = ttk.Combobox(patient_frame, values=["М", "Ж"], width=3)
        self.gender_combo.grid(row=3, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(patient_frame, text="Год рождения:").grid(row=4, column=0, sticky=tk.W)
        self.birth_year_entry = ttk.Entry(patient_frame, width=8)
        self.birth_year_entry.grid(row=4, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(patient_frame, text="Диагноз:").grid(row=5, column=0, sticky=tk.W)
        self.diagnosis_entry = ttk.Entry(patient_frame)
        self.diagnosis_entry.grid(row=5, column=1, sticky=tk.EW, padx=5, pady=2)

        # Кнопка анализа ЭКГ
        self.analyze_btn = ttk.Button(left_info_panel, text="Анализ ЭКГ", command=self.analyze_ecg)
        self.analyze_btn.pack(pady=10, fill=tk.X)

        # Добавляем текстовое поле для результатов
        results_frame = ttk.LabelFrame(left_info_panel, text="Результаты анализа")
        results_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)


        self.result_text = tk.Text(results_frame, height=10, wrap=tk.WORD, width=40)  # 40 символов в ширину
        scrollbar = ttk.Scrollbar(results_frame, command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text.pack(fill=tk.BOTH, expand=False)

        # Переносим остальные элементы управления в правую панель
        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Time Domain Tab
        time_frame = ttk.Frame(self.notebook)
        self.notebook.add(time_frame, text='Oscilloscope')

        # Control Panel
        control_frame = ttk.Frame(time_frame)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        left_panel = ttk.Frame(control_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.X, expand=True)

        right_panel = ttk.Frame(control_frame)
        right_panel.pack(side=tk.RIGHT)

        # Port settings
        ttk.Label(left_panel, text="Port:").pack(side=tk.LEFT)
        self.port_combo = ttk.Combobox(left_panel, width=15)
        self.port_combo.pack(side=tk.LEFT)

        ttk.Label(left_panel, text="Baud:").pack(side=tk.LEFT, padx=(10, 0))
        self.baud_combo = ttk.Combobox(left_panel, values=[9600, 19200, 38400, 57600, 115200, 250000, 500000, 1000000],
                                       width=10)
        self.baud_combo.current(6)
        self.baud_combo.pack(side=tk.LEFT)

        self.connect_btn = ttk.Button(left_panel, text="Connect", command=self.toggle_connection)
        self.connect_btn.pack(side=tk.LEFT, padx=10)

        # Sample settings
        ttk.Label(left_panel, text="Rate (Hz):").pack(side=tk.LEFT)
        self.rate_entry = ttk.Entry(left_panel, width=8)
        self.rate_entry.insert(0, "533")
        self.rate_entry.pack(side=tk.LEFT)
        self.rate_entry.bind("<FocusOut>", self.update_sample_rate)

        ttk.Label(left_panel, text="Points:").pack(side=tk.LEFT)
        self.points_entry = ttk.Entry(left_panel, width=8)
        self.points_entry.insert(0, "1000")
        self.points_entry.pack(side=tk.LEFT)

        # Filter settings
        ttk.Label(left_panel, text="Filter:").pack(side=tk.LEFT)
        self.filter_combo = ttk.Combobox(left_panel, values=['None', 'LPF', 'HPF', 'BPF'], width=6)
        self.filter_combo.current(0)
        self.filter_combo.pack(side=tk.LEFT)
        self.filter_combo.bind("<<ComboboxSelected>>", self.update_filter_ui)

        ttk.Label(left_panel, text="Order:").pack(side=tk.LEFT, padx=(10, 0))
        self.filter_order_combo = ttk.Combobox(left_panel, values=[1, 2, 4], width=3)
        self.filter_order_combo.current(0)
        self.filter_order_combo.pack(side=tk.LEFT)

        self.std_cutoff_frame = ttk.Frame(left_panel)
        self.std_cutoff_label = ttk.Label(self.std_cutoff_frame, text="Cutoff (Hz):")
        self.std_cutoff_label.pack(side=tk.LEFT)
        self.std_cutoff_entry = ttk.Entry(self.std_cutoff_frame, width=8)
        self.std_cutoff_entry.insert(0, "50")
        self.std_cutoff_entry.pack(side=tk.LEFT)
        self.std_cutoff_frame.pack(side=tk.LEFT)

        self.bpf_cutoff_frame = ttk.Frame(left_panel)
        self.bpf_low_label = ttk.Label(self.bpf_cutoff_frame, text="Low:")
        self.bpf_low_label.pack(side=tk.LEFT)
        self.bpf_low_entry = ttk.Entry(self.bpf_cutoff_frame, width=6)
        self.bpf_low_entry.insert(0, "20")
        self.bpf_low_entry.pack(side=tk.LEFT)
        self.bpf_high_label = ttk.Label(self.bpf_cutoff_frame, text="High:")
        self.bpf_high_label.pack(side=tk.LEFT)
        self.bpf_high_entry = ttk.Entry(self.bpf_cutoff_frame, width=6)
        self.bpf_high_entry.insert(0, "100")
        self.bpf_high_entry.pack(side=tk.LEFT)
        self.bpf_cutoff_frame.pack_forget()

        # Buttons
        self.export_btn = ttk.Button(right_panel, text="📤 Export", command=self.export_data)
        self.export_btn.pack(side=tk.RIGHT, padx=5)
        self.export_btn.state(['disabled'])

        self.pause_btn = ttk.Button(right_panel, text="⏸ Stop", command=self.toggle_pause)
        self.pause_btn.pack(side=tk.RIGHT, padx=5)
        self.pause_btn.state(['disabled'])

    def analyze_ecg(self):
        """Анализ ЭКГ сигнала с корректной визуализацией"""
        # Закрываем все предыдущие графики
        plt.close('all')

        file_path = filedialog.askopenfilename(
            filetypes=[("Excel files", "*.xlsx *.xls"), ("CSV files", "*.csv")],
            title="Выберите файл с ЭКГ данными"
        )

        if not file_path:
            return

        try:
            # Загрузка данных
            df = pd.read_excel(file_path)

            # Проверка наличия нужных столбцов
            required_columns = ['Time (s)', 'Ch1_Filtered']
            if not all(col in df.columns for col in required_columns):
                missing = [col for col in required_columns if col not in df.columns]
                messagebox.showerror("Ошибка", f"Отсутствуют необходимые столбцы: {', '.join(missing)}")
                return

            time_data = df['Time (s)'].values
            ecg_signal = df['Ch1_Filtered'].values

            # Проверка минимальной длины сигнала
            min_length = 250
            if len(ecg_signal) < min_length:
                messagebox.showwarning("Предупреждение",
                                       f"Слишком короткая запись ЭКГ. Минимум {min_length} точек, получено {len(ecg_signal)}")
                return

            # Оценка частоты дискретизации
            sampling_rate = 1 / np.mean(np.diff(time_data)) if len(time_data) > 1 else 250

            # Фильтрация сигнала для лучшего анализа
            ecg_filtered = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)

            # Детекция R-пиков
            try:
                _, rpeaks = nk.ecg_peaks(ecg_filtered, sampling_rate=sampling_rate)
                peaks = rpeaks['ECG_R_Peaks']

                if len(peaks) < 2:
                    messagebox.showwarning("Предупреждение",
                                           f"Обнаружено только {len(peaks)} R-пиков. Необходимо минимум 2 для анализа.")
                    return
            except Exception as e:
                messagebox.showerror("Ошибка детекции", f"Не удалось обнаружить R-пики: {str(e)}")
                return

            # Расчет показателей
            rr_intervals = np.diff(peaks) / sampling_rate
            heart_rate = 60 / np.mean(rr_intervals)

            # Подготовка текста с результатами
            result_text = (
                "=== Результаты анализа ЭКГ ===\n\n"
                f"Длительность записи: {time_data[-1] - time_data[0]:.1f} сек\n"
                f"Частота дискретизации: {sampling_rate:.1f} Гц\n"
                f"Общее количество точек: {len(ecg_signal)}\n"
                f"Обнаружено R-пиков: {len(peaks)}\n"
                f"Средняя ЧСС: {heart_rate:.1f} уд/мин\n"
                f"Диапазон ЧСС: {60 / rr_intervals.max():.1f}-{60 / rr_intervals.min():.1f} уд/мин\n\n"
            )

            # Проверка аритмии
            if np.std(rr_intervals) > 0.1:
                result_text += "ВНИМАНИЕ: Обнаружена возможная аритмия (высокая вариабельность RR-интервалов)\n"

            # Очистка и вывод результатов
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, result_text)

            # Создаем отдельное окно для графика ЭКГ
            ecg_fig = plt.figure("Анализ ЭКГ", figsize=(12, 6))

            # Основной график ЭКГ
            ax = ecg_fig.add_subplot(111)
            ax.plot(time_data, ecg_filtered, label='Фильтрованный ЭКГ сигнал')
            ax.scatter(time_data[peaks], ecg_filtered[peaks], color='red', label='R-пики')

            # Настройки графика
            ax.set_title(f"ЭКГ анализ (ЧСС: {heart_rate:.1f} уд/мин)")
            ax.set_xlabel("Время (с)")
            ax.set_ylabel("Амплитуда")
            ax.legend()
            ax.grid(True)

            plt.tight_layout()
            plt.show()

        except Exception as e:
            messagebox.showerror("Ошибка анализа", f"Не удалось проанализировать ЭКГ:\n{str(e)}")
            plt.close('all')


    def setup_plots(self):
        time_frame = self.notebook.winfo_children()[0]
        self.fig_time = plt.figure(figsize=(10, 8), dpi=100)
        self.axes = []
        self.lines = []

        for i in range(self.num_channels):
            ax = self.fig_time.add_subplot(self.num_channels, 1, i + 1)
            line, = ax.plot([], [], lw=1, color=self.channels[i]['color'])
            ax.set_ylabel(f'Ch {i + 1}')
            ax.grid(True)
            self.axes.append(ax)
            self.lines.append(line)

        self.annotations = []
        for i, ax in enumerate(self.axes):
            annotation = ax.annotate('',
                                     xy=(0, 0),
                                     xytext=(5, -15 if i < self.num_channels - 1 else 5),
                                     textcoords='offset points',
                                     bbox=dict(boxstyle="round", fc="w", alpha=0.8),
                                     arrowprops=dict(arrowstyle="->")
                                     )
            annotation.set_visible(False)
            self.annotations.append(annotation)

        self.fig_time.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.fig_time.canvas.mpl_connect('figure_leave_event', self.on_leave_figure)

        self.axes[-1].set_xlabel('Time (seconds)')
        self.canvas_time = FigureCanvasTkAgg(self.fig_time, master=time_frame)
        self.canvas_time.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def on_mouse_move(self, event):
        if event.inaxes is None:
            for ann in self.annotations:
                ann.set_visible(False)
            self.canvas_time.draw_idle()
            return

        for i, ax in enumerate(self.axes):
            if ax == event.inaxes:
                line = self.lines[i]
                xdata = line.get_xdata()
                ydata = line.get_ydata()

                if len(xdata) == 0:
                    continue

                idx = np.abs(xdata - event.xdata).argmin()
                x = xdata[idx]
                y = ydata[idx]

                self.annotations[i].xy = (x, y)
                self.annotations[i].set_text(f'x={x:.2f}s, y={y:.2f}mV')
                self.annotations[i].set_visible(True)
            else:
                self.annotations[i].set_visible(False)

        self.canvas_time.draw_idle()

    def on_leave_figure(self, event):
        for ann in self.annotations:
            ann.set_visible(False)
        self.canvas_time.draw_idle()

    def update_filter_ui(self, event=None):
        filter_type = self.filter_combo.get()
        if filter_type == 'BPF':
            self.std_cutoff_frame.pack_forget()
            self.bpf_cutoff_frame.pack(side=tk.LEFT)
        else:
            self.bpf_cutoff_frame.pack_forget()
            self.std_cutoff_frame.pack(side=tk.LEFT)

    def update_sample_rate(self, event=None):
        try:
            new_rate = float(self.rate_entry.get())
            if new_rate <= 0:
                raise ValueError
            self.sample_rate = new_rate
            self.sample_period = 1.0 / self.sample_rate
        except:
            messagebox.showerror("Error", "Invalid sample rate value!")
            self.rate_entry.delete(0, tk.END)
            self.rate_entry.insert(0, str(self.sample_rate))

    def refresh_ports(self):
        ports = [port.device for port in list_ports.comports()]
        self.port_combo['values'] = ports
        if ports:
            self.port_combo.current(0)

    def toggle_connection(self):
        if self.is_running:
            self.stop()
        else:
            self.start()

    def toggle_pause(self):
        if self.paused:
            self._reset_measurement()
            if self.ser and self.ser.is_open:
                self.ser.reset_input_buffer()
            self.paused = False
            self.pause_btn.config(text="⏸ Stop")
            self.start_time = time.time()
        else:
            self.paused = True
            self.pause_time = time.time()
            self.pause_btn.config(text="▶ Start")

    def _reset_measurement(self):
        self.total_points = 0
        self.timestamps.clear()
        self.data_queue.queue.clear()

        for ch in self.channels:
            ch['raw_data'].clear()
            ch['filtered_data'].clear()
            # Сбрасываем состояния фильтров
            ch['lpf_states'] = None
            ch['hpf_states'] = None
            ch['bpf_sections'] = None

        for line in self.lines:
            line.set_data([], [])
        self.canvas_time.draw()

    def start(self):
        try:
            self.buffer = bytearray()
            self.stop()
            self.total_points = 0
            self.timestamps.clear()

            self.buffer = bytearray()

            for ch in self.channels:
                ch['raw_data'].clear()
                ch['filtered_data'].clear()
                ch['lpf_states'] = None
                ch['hpf_states'] = None
                ch['bpf_sections'] = None

            self.ser = serial.Serial(
                self.port_combo.get(),
                int(self.baud_combo.get()),
                timeout=0.01
            )
            self.ser.reset_input_buffer()
            self.is_running = True
            self.paused = False
            self.pause_btn.state(['!disabled'])
            self.export_btn.state(['!disabled'])
            self.connect_btn.config(text="Disconnect")
            self.start_time = time.time()

            for line in self.lines:
                line.set_data([], [])
            self.canvas_time.draw()

            self.read_thread = threading.Thread(target=self.read_from_port, daemon=True)
            self.read_thread.start()
            self.update_plot()
        except Exception as e:
            print("Error:", e)
            messagebox.showerror("Connection Error", str(e))

    def read_from_port(self):
        HEADER1 = 0xAA
        HEADER2 = 0x55
        PACKET_SIZE = 19

        while self.is_running:
            if self.ser and self.ser.in_waiting:
                try:
                    raw_data = self.ser.read(self.ser.in_waiting)
                    self.buffer.extend(raw_data)

                    while len(self.buffer) >= PACKET_SIZE:
                        start_idx = -1
                        for i in range(len(self.buffer) - 1):
                            if self.buffer[i] == HEADER1 and self.buffer[i + 1] == HEADER2:
                                start_idx = i
                                break

                        if start_idx < 0:
                            self.buffer.clear()
                            break

                        if len(self.buffer) - start_idx < PACKET_SIZE:
                            del self.buffer[:start_idx]
                            break

                        packet = self.buffer[start_idx:start_idx + PACKET_SIZE]
                        del self.buffer[:start_idx + PACKET_SIZE]

                        if packet[0] != HEADER1 or packet[1] != HEADER2:
                            continue

                        data_part = packet[2:18]
                        received_checksum = packet[18]

                        computed_checksum = 0
                        for byte in packet[:18]:
                            computed_checksum ^= byte

                        if computed_checksum != received_checksum:
                            print(f"CRC error: {computed_checksum} vs {received_checksum}")
                            continue

                        try:
                            int_values = struct.unpack('<4i', data_part)
                            #'<4i' для 4 целых чисел(int32) '<4f' для 4 чисел с плавающей точкой(float32)
                            values = [float(val) for val in int_values]
                            self.data_queue.put(values)
                        except struct.error as e:
                            print(f"Unpack error: {e}")

                    if len(self.buffer) > 1024:
                        self.buffer = bytearray()

                except Exception as e:
                    print("Read error:", e)

    def apply_filter(self, values):

        filter_type = self.filter_combo.get()
        filtered = []

        try:
            order = int(self.filter_order_combo.get())
        except:
            order = 1

        for i, value in enumerate(values):
            ch = self.channels[i]

            if filter_type == 'LPF':
                try:
                    cutoff = float(self.std_cutoff_entry.get())
                except:
                    cutoff = 50.0

                alpha = 1 / (1 + 1 / (2 * np.pi * cutoff / self.sample_rate))

                # Инициализация состояний при необходимости
                if ch['lpf_states'] is None or len(ch['lpf_states']) != order:
                    ch['lpf_states'] = [0.0] * order

                # Применение каскада фильтров
                current_value = value
                for j in range(order):
                    state = ch['lpf_states'][j]
                    filtered_val = alpha * current_value + (1 - alpha) * state
                    ch['lpf_states'][j] = filtered_val
                    current_value = filtered_val

                filtered.append(filtered_val)

            elif filter_type == 'HPF':
                try:
                    cutoff = float(self.std_cutoff_entry.get())
                except:
                    cutoff = 50.0

                alpha = 1 / (1 + 1 / (2 * np.pi * cutoff / self.sample_rate))

                # Инициализация состояний при необходимости
                if ch['hpf_states'] is None or len(ch['hpf_states']) != order:
                    ch['hpf_states'] = [{'state': 0.0, 'prev_input': 0.0} for _ in range(order)]

                # Применение каскада фильтров
                current_value = value
                for j in range(order):
                    state_dict = ch['hpf_states'][j]
                    hp = alpha * (state_dict['state'] + current_value - state_dict['prev_input'])
                    state_dict['state'] = hp
                    state_dict['prev_input'] = current_value
                    current_value = hp

                filtered.append(hp)

            elif filter_type == 'BPF':
                try:
                    cutoff_low = float(self.bpf_low_entry.get())
                    cutoff_high = float(self.bpf_high_entry.get())
                except:
                    cutoff_low = 20.0
                    cutoff_high = 100.0

                if cutoff_low > cutoff_high:
                    cutoff_low, cutoff_high = cutoff_high, cutoff_low

                alpha_low = 1 / (1 + 1 / (2 * np.pi * cutoff_low / self.sample_rate))
                alpha_high = 1 / (1 + 1 / (2 * np.pi * cutoff_high / self.sample_rate))

                # Инициализация секций при необходимости
                if ch['bpf_sections'] is None or len(ch['bpf_sections']) != order:
                    ch['bpf_sections'] = [{'lpf': 0.0, 'hpf': 0.0, 'prev_lpf': 0.0} for _ in range(order)]

                # Применение каскада секций BPF
                current_value = value
                for j in range(order):
                    section = ch['bpf_sections'][j]

                    # LPF часть
                    lpf_val = alpha_high * current_value + (1 - alpha_high) * section['lpf']
                    section['lpf'] = lpf_val

                    # HPF часть
                    hpf_val = alpha_low * (section['hpf'] + lpf_val - section['prev_lpf'])
                    section['hpf'] = hpf_val
                    section['prev_lpf'] = lpf_val

                    current_value = hpf_val

                filtered.append(hpf_val)

            else:
                filtered.append(value)

        return filtered

    def update_plot(self):
        if self.is_running and not self.paused:
            try:
                max_points = 200
                processed = 0

                while not self.data_queue.empty() and processed < max_points:
                    raw_values = self.data_queue.get_nowait()
                    current_time = self.total_points * self.sample_period

                    self.total_points += 1

                    filtered_values = self.apply_filter(raw_values)

                    self.timestamps.append(current_time)
                    for i in range(self.num_channels):
                        self.channels[i]['raw_data'].append(raw_values[i])
                        self.channels[i]['filtered_data'].append(filtered_values[i])
                    processed += 1

                points_to_show = min(int(self.points_entry.get()), len(self.timestamps))
                if points_to_show > 0:
                    time_axis = np.array(self.timestamps)[-points_to_show:]

                    for i, ax in enumerate(self.axes):
                        data_axis = np.array(self.channels[i]['filtered_data'])[-points_to_show:]
                        self.lines[i].set_data(time_axis, data_axis)

                        ax.relim()
                        ax.autoscale_view(scalex=False, scaley=True)

                    if len(time_axis) > 1:
                        x_min = max(0, time_axis[0])
                        x_max = time_axis[-1] + 0.1 * (time_axis[-1] - time_axis[0])
                        if x_min == x_max:
                            x_min -= 0.1
                            x_max += 0.1
                    elif len(time_axis) == 1:
                        x_min = time_axis[0] - 0.5
                        x_max = time_axis[0] + 0.5
                    else:
                        x_min, x_max = 0, 1

                    for ax in self.axes:
                        ax.set_xlim(x_min, x_max)

                    self.canvas_time.draw()

            except Exception as e:
                print(f"Update error: {e}")

        self.after_id = self.master.after(5, self.update_plot)

    def increase_scale(self, event=None):
        for ax in self.axes:
            y_center = (ax.get_ylim()[0] + ax.get_ylim()[1]) / 2
            y_half_range = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.4
            ax.set_ylim(y_center - y_half_range, y_center + y_half_range)
        self.canvas_time.draw()

    def decrease_scale(self, event=None):
        for ax in self.axes:
            y_center = (ax.get_ylim()[0] + ax.get_ylim()[1]) / 2
            y_half_range = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.625
            ax.set_ylim(y_center - y_half_range, y_center + y_half_range)
        self.canvas_time.draw()

    def export_data(self):
        try:
            patient_info = {
                "Фамилия": self.surname_entry.get(),
                "Имя": self.name_entry.get(),
                "Отчество": self.patronymic_entry.get(),
                "Пол": self.gender_combo.get(),
                "Год рождения": self.birth_year_entry.get(),
                "Диагноз": self.diagnosis_entry.get(),
                "Частота дискретизации (Гц)": self.sample_rate,
                "Дата записи": time.strftime("%Y-%m-%d %H:%M:%S")
            }

            try:
                points_to_save = int(self.points_entry.get())
                if points_to_save <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Error", "Invalid points value! Please enter a positive integer.")
                return

            if not self.timestamps:
                messagebox.showwarning("No Data", "There is no data to export.")
                return

            points_available = min(points_to_save, len(self.timestamps))
            if points_available == 0:
                messagebox.showwarning("No Data", "There is no data to export.")
                return

            data_dict = {'Time (s)': list(self.timestamps)[-points_available:]}
            for i in range(self.num_channels):
                data_dict[f'Ch{i + 1}_Raw'] = list(self.channels[i]['raw_data'])[-points_available:]
                data_dict[f'Ch{i + 1}_Filtered'] = list(self.channels[i]['filtered_data'])[-points_available:]

            df_data = pd.DataFrame(data_dict)
            df_info = pd.DataFrame(list(patient_info.items()), columns=['Parameter', 'Value'])

            file_path = filedialog.asksaveasfilename(
                defaultextension='.xlsx',
                filetypes=[('Excel files', '*.xlsx'), ('CSV files', '*.csv')],
                title="Save data as"
            )

            if not file_path:
                return

            if file_path.endswith('.csv'):
                messagebox.showinfo("Success", f"Data exported successfully to:\n{file_path}")
            else:
                try:
                    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                        df_info.to_excel(writer,
                                         sheet_name='Patient Info',
                                         index=False,
                                         header=['Parameter', 'Value'])

                        df_data.to_excel(writer,
                                         sheet_name='Measurement Data',
                                         index=False)

                        from openpyxl.utils import get_column_letter

                        worksheet_info = writer.sheets['Patient Info']
                        for idx in range(len(df_info.columns)):
                            col_letter = get_column_letter(idx + 1)
                            max_len = max(
                                df_info.iloc[:, idx].astype(str).apply(len).max(),
                                len(str(df_info.columns[idx]))
                            )
                            worksheet_info.column_dimensions[col_letter].width = max_len + 2

                        worksheet_data = writer.sheets['Measurement Data']
                        for idx in range(len(df_data.columns)):
                            col_letter = get_column_letter(idx + 1)
                            max_len = max(
                                df_data.iloc[:, idx].astype(str).apply(len).max(),
                                len(str(df_data.columns[idx]))
                            )
                            worksheet_data.column_dimensions[col_letter].width = max_len + 2

                    messagebox.showinfo("Success", f"Data exported successfully to:\n{file_path}")

                except Exception as e:
                    messagebox.showerror("Export Error", f"Failed to export data to Excel:\n{str(e)}")

        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export data:\n{str(e)}")

    def stop(self):
        if self.is_running or self.ser is not None:
            self.is_running = False

            if self.after_id:
                self.master.after_cancel(self.after_id)
                self.after_id = None

            if self.ser and self.ser.is_open:
                try:
                    self.ser.close()
                    print("COM port closed successfully")
                except serial.SerialException as e:
                    print(f"Error closing port: {str(e)}")
                finally:
                    self.ser = None

            if hasattr(self, 'read_thread'):
                try:
                    self.read_thread.join(timeout=0.5)
                    if self.read_thread.is_alive():
                        print("Warning: Read thread not terminated properly")
                except Exception as e:
                    print(f"Thread join error: {str(e)}")

            self.connect_btn.config(text="Connect")
            self.pause_btn.state(['disabled'])
            self.export_btn.state(['disabled'])

        self.data_queue.queue.clear()
        self.timestamps.clear()
        self.buffer = bytearray()
        for ch in self.channels:
            ch['raw_data'].clear()
            ch['filtered_data'].clear()


if __name__ == "__main__":
    root = tk.Tk()
    root.title("4-Channel Oscilloscope")
    root.geometry("1200x800")
    app = Oscilloscope(root)
    root.mainloop()
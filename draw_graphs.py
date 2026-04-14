import numpy as np
import matplotlib.pyplot as plt

# Ваши данные из экспериментов
sizes = [200, 400, 800, 1200, 1600, 2000]
times = [8.506, 62.633, 535.114, 2006.269, 6161.557, 5967.706]

# Теоретическая кривая O(N^3)
n_ref = sizes[0]
t_ref = times[0]
theoretical = t_ref * (np.array(sizes) / n_ref) ** 3

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(sizes, times, 'bo-', linewidth=2, markersize=8, label='Эксперимент')
plt.plot(sizes, theoretical, 'r--', linewidth=2, label='Теория O(N³)')

plt.xlabel('Размер матрицы N', fontsize=12)
plt.ylabel('Время выполнения (мс)', fontsize=12)
plt.title('Зависимость времени умножения матриц от размера', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

# Сохраняем и показываем
plt.savefig('graph.png', dpi=150, bbox_inches='tight')
plt.show()

# Вывод таблицы
print("\n" + "="*50)
print("РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТОВ")
print("="*50)
print(f"{'N':>6} | {'Время (мс)':>12}")
print("-"*30)
for n, t in zip(sizes, times):
    print(f"{n:6d} | {t:12.3f}")
print("="*50)
print("\n✓ График сохранён как 'graph.png'")
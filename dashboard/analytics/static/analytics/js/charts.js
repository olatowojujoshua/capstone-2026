/* ═══════════════════════════════════════════════
   Chart.js Helper Functions — Dark Theme
   ═══════════════════════════════════════════════ */

/**
 * Returns shared defaults for Chart.js charts.
 */
function chartDefaults() {
    return {
        responsive: true,
        maintainAspectRatio: false,
        animation: {
            duration: 800,
            easing: 'easeOutQuart',
        },
        plugins: {
            legend: {
                display: false,
                labels: {
                    color: 'rgba(255,255,255,0.6)',
                    font: { family: 'Inter', size: 12 },
                    padding: 16,
                    usePointStyle: true,
                    pointStyleWidth: 10,
                }
            },
            tooltip: {
                backgroundColor: 'rgba(15, 15, 26, 0.95)',
                titleColor: '#f1f5f9',
                bodyColor: 'rgba(255,255,255,0.7)',
                borderColor: 'rgba(99, 102, 241, 0.3)',
                borderWidth: 1,
                cornerRadius: 10,
                padding: 12,
                titleFont: { family: 'Inter', weight: '600', size: 13 },
                bodyFont: { family: 'Inter', size: 12 },
                displayColors: true,
                boxPadding: 4,
            }
        },
        scales: {
            x: {
                grid: {
                    color: 'rgba(255,255,255,0.04)',
                    drawBorder: false,
                },
                ticks: {
                    color: 'rgba(255,255,255,0.4)',
                    font: { family: 'Inter', size: 11 },
                    maxRotation: 0,
                },
                border: { display: false }
            },
            y: {
                grid: {
                    color: 'rgba(255,255,255,0.04)',
                    drawBorder: false,
                },
                ticks: {
                    color: 'rgba(255,255,255,0.4)',
                    font: { family: 'Inter', size: 11 },
                },
                border: { display: false }
            }
        }
    };
}

/**
 * Generate gradient colors for bar charts.
 */
function createGradientColors(count, palette) {
    const palettes = {
        indigo: { r: 99, g: 102, b: 241 },
        cyan: { r: 6, g: 182, b: 212 },
        green: { r: 16, g: 185, b: 129 },
        red: { r: 244, g: 63, b: 94 },
        purple: { r: 168, g: 85, b: 247 },
    };
    const base = palettes[palette] || palettes.indigo;
    const colors = [];
    for (let i = 0; i < count; i++) {
        const factor = 0.6 + (0.4 * i / Math.max(count - 1, 1));
        colors.push(`rgba(${base.r}, ${base.g}, ${base.b}, ${factor.toFixed(2)})`);
    }
    return colors;
}

/**
 * Create a line chart.
 */
function createLineChart(canvasId, labels, data, label, borderColor, bgColor, showPoints) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    const gradient = ctx.createLinearGradient(0, 0, 0, ctx.canvas.clientHeight);
    gradient.addColorStop(0, bgColor || 'rgba(99, 102, 241, 0.15)');
    gradient.addColorStop(1, 'rgba(99, 102, 241, 0)');

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: label,
                data: data,
                borderColor: borderColor,
                backgroundColor: gradient,
                fill: true,
                tension: 0.4,
                pointRadius: showPoints !== false ? 3 : 0,
                pointBackgroundColor: borderColor,
                pointBorderColor: 'transparent',
                pointHoverRadius: 6,
                pointHoverBackgroundColor: borderColor,
                pointHoverBorderColor: '#fff',
                pointHoverBorderWidth: 2,
                borderWidth: 2.5,
            }]
        },
        options: chartDefaults()
    });
}

/**
 * Create a vertical bar chart.
 */
function createBarChart(canvasId, labels, data, label, colors) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: label,
                data: data,
                backgroundColor: Array.isArray(colors) ? colors : colors,
                borderColor: 'transparent',
                borderWidth: 0,
                borderRadius: 6,
                borderSkipped: false,
                maxBarThickness: 48,
            }]
        },
        options: chartDefaults()
    });
}

/**
 * Create a horizontal bar chart.
 */
function createHorizontalBarChart(canvasId, labels, data, label, color) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: label,
                data: data,
                backgroundColor: color,
                borderColor: 'transparent',
                borderWidth: 0,
                borderRadius: 4,
                borderSkipped: false,
            }]
        },
        options: {
            ...chartDefaults(),
            indexAxis: 'y',
            scales: {
                x: {
                    ...chartDefaults().scales.x,
                },
                y: {
                    ...chartDefaults().scales.y,
                    ticks: {
                        ...chartDefaults().scales.y.ticks,
                        font: { family: 'Inter', size: 10 },
                    }
                }
            }
        }
    });
}

/**
 * Create a doughnut chart.
 */
function createDoughnutChart(canvasId, labels, data) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    const colors = [
        'rgba(99, 102, 241, 0.85)',
        'rgba(6, 182, 212, 0.85)',
        'rgba(16, 185, 129, 0.85)',
        'rgba(249, 115, 22, 0.85)',
        'rgba(244, 63, 94, 0.85)',
        'rgba(168, 85, 247, 0.85)',
        'rgba(234, 179, 8, 0.85)',
    ];

    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: colors.slice(0, data.length),
                borderColor: 'rgba(10, 10, 20, 0.8)',
                borderWidth: 2,
                hoverOffset: 8,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '62%',
            animation: { animateRotate: true, duration: 1000 },
            plugins: {
                legend: {
                    position: 'right',
                    labels: {
                        color: 'rgba(255,255,255,0.6)',
                        font: { family: 'Inter', size: 12 },
                        padding: 14,
                        usePointStyle: true,
                        pointStyleWidth: 10,
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(15, 15, 26, 0.95)',
                    titleColor: '#f1f5f9',
                    bodyColor: 'rgba(255,255,255,0.7)',
                    borderColor: 'rgba(99, 102, 241, 0.3)',
                    borderWidth: 1,
                    cornerRadius: 10,
                    padding: 12,
                    callbacks: {
                        label: function(context) {
                            return ` ${context.label}: $${context.parsed.toFixed(2)}`;
                        }
                    }
                }
            }
        }
    });
}

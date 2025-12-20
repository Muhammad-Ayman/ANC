/**
 * Audio Visualization Module
 * Handles waveform rendering and convergence plots
 */

class AudioVisualizer {
    constructor() {
        this.colors = {
            primary: '#6366f1',
            secondary: '#8b5cf6',
            tertiary: '#ec4899',
            background: '#121826',
            grid: '#1f2937',
            text: '#9ca3af'
        };
    }

    /**
     * Draw waveform on canvas
     */
    drawWaveform(canvas, waveformData, color = this.colors.primary) {
        if (!canvas || !waveformData || waveformData.length === 0) return;

        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;

        // Clear canvas
        ctx.fillStyle = this.colors.background;
        ctx.fillRect(0, 0, width, height);

        // Draw center line
        ctx.strokeStyle = this.colors.grid;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, height / 2);
        ctx.lineTo(width, height / 2);
        ctx.stroke();

        // Draw waveform
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();

        const step = width / waveformData.length;

        waveformData.forEach((point, i) => {
            const x = i * step;
            const y = (height / 2) - (point.amplitude * height / 2);

            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });

        ctx.stroke();

        // Add glow effect
        ctx.shadowBlur = 10;
        ctx.shadowColor = color;
        ctx.stroke();
        ctx.shadowBlur = 0;
    }

    /**
     * Draw comparison of two waveforms
     */
    drawComparison(canvas, beforeData, afterData) {
        if (!canvas || !beforeData || !afterData) return;

        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;

        // Clear canvas
        ctx.fillStyle = this.colors.background;
        ctx.fillRect(0, 0, width, height);

        // Draw grid lines
        this.drawGrid(ctx, width, height);

        // Draw 'before' waveform (semi-transparent)
        this.drawWaveformPath(ctx, beforeData, width, height, this.colors.tertiary, 0.5);

        // Draw 'after' waveform
        this.drawWaveformPath(ctx, afterData, width, height, this.colors.primary, 1.0);

        // Add legend
        this.drawLegend(ctx, width, height);
    }

    drawWaveformPath(ctx, data, width, height, color, alpha) {
        ctx.strokeStyle = color;
        ctx.globalAlpha = alpha;
        ctx.lineWidth = 2;
        ctx.beginPath();

        const step = width / data.length;

        data.forEach((point, i) => {
            const x = i * step;
            const y = (height / 2) - (point.amplitude * height / 2.2);

            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });

        ctx.stroke();
        ctx.globalAlpha = 1.0;
    }

    drawGrid(ctx, width, height) {
        ctx.strokeStyle = this.colors.grid;
        ctx.lineWidth = 1;

        // Horizontal center line
        ctx.beginPath();
        ctx.moveTo(0, height / 2);
        ctx.lineTo(width, height / 2);
        ctx.stroke();

        // Vertical grid lines (every 20%)
        for (let i = 1; i < 5; i++) {
            ctx.beginPath();
            ctx.moveTo(width * i / 5, 0);
            ctx.lineTo(width * i / 5, height);
            ctx.stroke();
        }
    }

    drawLegend(ctx, width, height) {
        const legendX = width - 150;
        const legendY = 20;

        // Before (original)
        ctx.fillStyle = this.colors.tertiary;
        ctx.fillRect(legendX, legendY, 30, 3);
        ctx.fillStyle = this.colors.text;
        ctx.font = '12px Inter';
        ctx.fillText('Before', legendX + 35, legendY + 4);

        // After (cleaned)
        ctx.fillStyle = this.colors.primary;
        ctx.fillRect(legendX, legendY + 20, 30, 3);
        ctx.fillStyle = this.colors.text;
        ctx.fillText('After', legendX + 35, legendY + 24);
    }

    /**
     * Draw convergence plot
     */
    drawConvergence(canvas, convergenceData) {
        if (!canvas || !convergenceData || convergenceData.length === 0) return;

        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;

        // Clear canvas
        ctx.fillStyle = this.colors.background;
        ctx.fillRect(0, 0, width, height);

        // Add padding
        const padding = { top: 40, right: 40, bottom: 50, left: 70 };
        const plotWidth = width - padding.left - padding.right;
        const plotHeight = height - padding.top - padding.bottom;

        // Find min/max values
        const values = convergenceData.map(d => d.value);
        const minValue = Math.min(...values);
        const maxValue = Math.max(...values);
        const times = convergenceData.map(d => d.time);
        const maxTime = Math.max(...times);

        // Draw axes
        this.drawAxes(ctx, padding, plotWidth, plotHeight);

        // Draw labels
        this.drawConvergenceLabels(ctx, padding, plotWidth, plotHeight, minValue, maxValue, maxTime);

        // Draw convergence line
        ctx.strokeStyle = this.colors.primary;
        ctx.lineWidth = 2.5;
        ctx.beginPath();

        convergenceData.forEach((point, i) => {
            const x = padding.left + (point.time / maxTime) * plotWidth;
            const y = padding.top + plotHeight - ((point.value - minValue) / (maxValue - minValue)) * plotHeight;

            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });

        ctx.stroke();

        // Add glow
        ctx.shadowBlur = 15;
        ctx.shadowColor = this.colors.primary;
        ctx.stroke();
        ctx.shadowBlur = 0;

        // Add title
        ctx.fillStyle = this.colors.text;
        ctx.font = 'bold 16px Inter';
        ctx.textAlign = 'center';
        ctx.fillText('Running RMS Convergence', width / 2, 25);
    }

    drawAxes(ctx, padding, plotWidth, plotHeight) {
        ctx.strokeStyle = this.colors.text;
        ctx.lineWidth = 2;

        // Y-axis
        ctx.beginPath();
        ctx.moveTo(padding.left, padding.top);
        ctx.lineTo(padding.left, padding.top + plotHeight);
        ctx.stroke();

        // X-axis
        ctx.beginPath();
        ctx.moveTo(padding.left, padding.top + plotHeight);
        ctx.lineTo(padding.left + plotWidth, padding.top + plotHeight);
        ctx.stroke();
    }

    drawConvergenceLabels(ctx, padding, plotWidth, plotHeight, minValue, maxValue, maxTime) {
        ctx.fillStyle = this.colors.text;
        ctx.font = '12px Inter';

        // Y-axis label
        ctx.save();
        ctx.translate(15, padding.top + plotHeight / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.textAlign = 'center';
        ctx.fillText('RMS Value', 0, 0);
        ctx.restore();

        // X-axis label
        ctx.textAlign = 'center';
        ctx.fillText('Time (seconds)', padding.left + plotWidth / 2, padding.top + plotHeight + 35);

        // Y-axis ticks
        ctx.textAlign = 'right';
        for (let i = 0; i <= 5; i++) {
            const value = minValue + (maxValue - minValue) * (i / 5);
            const y = padding.top + plotHeight - (i / 5) * plotHeight;
            ctx.fillText(value.toFixed(4), padding.left - 10, y + 4);
        }

        // X-axis ticks
        ctx.textAlign = 'center';
        for (let i = 0; i <= 5; i++) {
            const value = (maxTime * i / 5).toFixed(1);
            const x = padding.left + (i / 5) * plotWidth;
            ctx.fillText(value, x, padding.top + plotHeight + 20);
        }
    }

    /**
     * Initialize canvas with proper dimensions
     */
    initCanvas(canvas) {
        if (!canvas) return;

        const rect = canvas.getBoundingClientRect();
        canvas.width = rect.width * window.devicePixelRatio;
        canvas.height = rect.height * window.devicePixelRatio;

        const ctx = canvas.getContext('2d');
        ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

        canvas.style.width = rect.width + 'px';
        canvas.style.height = rect.height + 'px';
    }
}

// Export for use in app.js
window.AudioVisualizer = AudioVisualizer;

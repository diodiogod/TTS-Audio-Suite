export function bindEmotionRadialPicker(button, emotions, onPick) {
    button.style.background = `linear-gradient(90deg, ${emotions.map(emotion => emotion.color).join(", ")})`;
    const angleStep = 2 * Math.PI / emotions.length;

    button.addEventListener("pointerdown", event => {
        if (event.button !== 0 || !event.isPrimary) return;
        event.preventDefault();
        event.stopPropagation();

        const size = 220;
        const center = size / 2;
        const maxRadius = 82;
        const { clientX: originX, clientY: originY, pointerId } = event;
        const canvas = document.createElement("canvas");
        canvas.width = canvas.height = size;
        Object.assign(canvas.style, {
            position: "fixed", left: `${originX - center}px`, top: `${originY - center}px`,
            width: `${size}px`, height: `${size}px`, zIndex: "100002", pointerEvents: "none",
            borderRadius: "50%", filter: "drop-shadow(0 8px 20px rgba(0,0,0,.85))",
        });
        document.body.appendChild(canvas);
        const ctx = canvas.getContext("2d");
        let selectedIndex = 0;
        let magnitude = 0;

        const draw = () => {
            ctx.clearRect(0, 0, size, size);
            const { name, color } = emotions[selectedIndex];
            ctx.fillStyle = "rgba(25,25,27,.96)";
            ctx.beginPath();
            ctx.arc(center, center, 104, 0, 2 * Math.PI);
            ctx.fill();
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.stroke();

            const selectedAngle = selectedIndex * angleStep - Math.PI / 2;
            ctx.fillStyle = `${color}38`;
            ctx.beginPath();
            ctx.moveTo(center, center);
            ctx.arc(center, center, 100, selectedAngle - angleStep / 2, selectedAngle + angleStep / 2);
            ctx.closePath();
            ctx.fill();

            emotions.forEach((emotion, index) => {
                const angle = index * angleStep - Math.PI / 2;
                ctx.fillStyle = index === selectedIndex ? emotion.color : "#aaa";
                ctx.font = index === selectedIndex ? "bold 11px Arial" : "10px Arial";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText(emotion.label ?? emotion.name, center + Math.cos(angle) * maxRadius, center + Math.sin(angle) * maxRadius);
            });

            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(center, center, 28, 0, 2 * Math.PI);
            ctx.fill();
            ctx.fillStyle = name === "happy" ? "#201b00" : "#f5f5f5";
            ctx.font = "bold 12px Arial";
            ctx.textAlign = "center";
            ctx.fillText(magnitude.toFixed(2), center, center);
        };

        const update = pointerEvent => {
            if (pointerEvent.pointerId !== pointerId) return;
            const dx = pointerEvent.clientX - originX;
            const dy = pointerEvent.clientY - originY;
            let angle = Math.atan2(dy, dx) + Math.PI / 2;
            if (angle < 0) angle += 2 * Math.PI;
            selectedIndex = Math.round(angle / angleStep) % emotions.length;
            const { name, min = 0, max, step } = emotions[selectedIndex];
            magnitude = min + Math.min(1, Math.hypot(dx, dy) / maxRadius) * (max - min);
            if (step) magnitude = Math.min(max, min + Math.round((magnitude - min) / step) * step);
            onPick(name, magnitude, false);
            draw();
        };

        const cleanup = () => {
            window.removeEventListener("pointermove", update, true);
            window.removeEventListener("pointerup", finish, true);
            window.removeEventListener("pointercancel", cancel, true);
            canvas.remove();
        };
        const finish = pointerEvent => {
            if (pointerEvent.pointerId !== pointerId) return;
            update(pointerEvent);
            cleanup();
            onPick(emotions[selectedIndex].name, magnitude, true);
        };
        const cancel = pointerEvent => {
            if (pointerEvent.pointerId === pointerId) cleanup();
        };
        window.addEventListener("pointermove", update, true);
        window.addEventListener("pointerup", finish, true);
        window.addEventListener("pointercancel", cancel, true);
        draw();
    });
}

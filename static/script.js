document.getElementById('prediction-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    const formData = new FormData(event.target);
    const data = Object.fromEntries(formData);

    // Kiểm tra restecg
    if (!data.restecg) {
        alert("Vui lòng chọn kết quả điện tâm đồ (0-2)!");
        return;
    }

    // Kiểm tra thalch
    const thalchValue = parseFloat(data.thalch);
    if (isNaN(thalchValue) || thalchValue < 70 || thalchValue > 200) {
        alert("Nhịp tim tối đa phải từ 70 đến 200!");
        return;
    }

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        });
        const result = await response.json();
        document.getElementById('recommendation').style.display = 'block';
        document.getElementById('recommendation-text').textContent = result.result;
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('recommendation').style.display = 'block';
        document.getElementById('recommendation-text').textContent = 'Có lỗi xảy ra, vui lòng thử lại.';
    }
});
document.getElementById('fraudForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const btn = document.getElementById('analyzeBtn');
    const resultCard = document.getElementById('resultCard');
    const originalText = btn.innerText;
    
    // UI Loading State
    btn.innerText = 'Analyzing...';
    btn.disabled = true;
    
    // Gather data
    const formData = new FormData(this);
    const data = Object.fromEntries(formData.entries());
    
    // Convert types
    data.amount = parseFloat(data.amount);
    data.oldbalanceOrg = parseFloat(data.oldbalanceOrg);
    data.newbalanceOrig = parseFloat(data.newbalanceOrig);
    data.hour = parseInt(data.hour);
    data.num_tx_24h = parseInt(data.num_tx_24h);
    data.location_consistency = parseInt(data.location_consistency);
    data.device_status = parseInt(data.device_status);
    data.account_age_days = parseInt(data.account_age_days);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            throw new Error('Analysis failed');
        }

        const result = await response.json();
        updateUI(result);
        
    } catch (error) {
        alert('Error: ' + error.message);
    } finally {
        btn.innerText = originalText;
        btn.disabled = false;
        resultCard.style.opacity = '1';
        resultCard.style.pointerEvents = 'all';
    }
});

function updateUI(result) {
    const probRing = document.getElementById('probRing');
    const probValue = document.getElementById('probValue');
    const statusBadge = document.getElementById('statusBadge');
    const resultDetails = document.getElementById('resultDetails');
    
    const probability = result.probability * 100;
    const color = probability > 70 ? '#ef4444' : (probability > 30 ? '#f59e0b' : '#22c55e');
    
    // Animate Ring
    probRing.style.background = `conic-gradient(${color} ${probability}%, var(--border) ${probability}% 100%)`;
    probValue.innerText = probability.toFixed(1) + '%';
    probValue.style.color = color;
    
    // Update Badge
    statusBadge.innerText = result.risk_level + " RISK";
    statusBadge.style.backgroundColor = color + '20'; // 20% opacity
    statusBadge.style.color = color;
    
    // Update Details
    if (result.is_fraud) {
        resultDetails.innerText = "This transaction fits high-risk fraud patterns. Immediate verification recommended.";
    } else {
        resultDetails.innerText = "This transaction appears legitimate based on historical patterns.";
    }
}

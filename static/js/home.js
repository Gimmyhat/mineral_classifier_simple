document.addEventListener('DOMContentLoaded', function() {
    const searchForm = document.getElementById('searchForm');
    if (searchForm) {
        searchForm.addEventListener('submit', async function(event) {
            event.preventDefault();
            await classifyMineral(event);
        });
    }
});

async function classifyMineral(event) {
    const form = event.target;
    const formData = new FormData(form);

    try {
        const response = await fetch('/classify', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Ошибка при получении данных');
        }

        const result = await response.json();
        displayResult(result);
    } catch (error) {
        console.error('Error:', error);
        displayError(error.message);
    }
}

function displayResult(result) {
    const resultDiv = document.getElementById('result');
    const resultTable = document.getElementById('resultTable');
    
    resultTable.innerHTML = `
        <tr>
            <td>Нормализованное название:</td>
            <td>${result.normalized_name_for_display}</td>
        </tr>
        <tr>
            <td>Название ГБЗ/ТБЗ:</td>
            <td>${result.pi_name_gbz_tbz}</td>
        </tr>
        <tr>
            <td>Группа недр:</td>
            <td>${result.pi_group_is_nedra}</td>
        </tr>
        <tr>
            <td>Единица измерения:</td>
            <td>${result.pi_measurement_unit}</td>
        </tr>
        <tr>
            <td>Альтернативная единица измерения:</td>
            <td>${result.pi_measurement_unit_alternative || '-'}</td>
        </tr>
    `;
    
    resultDiv.style.display = 'block';
}

function displayError(message) {
    const resultDiv = document.getElementById('result');
    const resultTable = document.getElementById('resultTable');
    
    resultTable.innerHTML = `
        <tr>
            <td colspan="2" class="error-message">
                Произошла ошибка при классификации: ${message}
            </td>
        </tr>
    `;
    
    resultDiv.style.display = 'block';
} 
const API_BASE_URL = 'http://localhost:8000';

// Global state
let currentKpiData = null;
let currentView = 'dashboard';
let currentBorrowerId = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM Content Loaded at', new Date().toLocaleTimeString());
    updateCurrentDate();
    setupEventListeners();

    // Try to fetch latest data from server first
    fetchDashboardData();

    // Recovery data from storage if it exists (Data in Local; View in Session)
    const savedData = localStorage.getItem('finance_data');
    const savedView = sessionStorage.getItem('current_view');
    const savedBorrowerId = sessionStorage.getItem('current_borrower_id');

    console.log('Storage Check:', {
        hasData: !!savedData,
        view: savedView,
        borrower: savedBorrowerId
    });

    if (savedData) {
        console.log('🔄 Attempting to recover data...');
        try {
            const data = JSON.parse(savedData);
            if (data && data.kpis) {
                currentKpiData = data;
                updateDashboard(data);
                console.log('Data successfully recovered from session storage');

                // Restore the previous view
                if (savedView === 'summary-details') {
                    const savedPeriod = sessionStorage.getItem('current_period_key');
                    if (savedPeriod) {
                        console.log('Restoring Summary Details List view for:', savedPeriod);
                        showSummaryDetailsListView(savedPeriod);
                    }
                }
            }
        } catch (e) {
            console.error('❌ Failed to parse saved data', e);
            localStorage.removeItem('finance_data');
        }
    } else {
        console.log('ℹ️ No saved data found in localStorage');
    }
});

// Update current date
function updateCurrentDate() {
    const dateElement = document.getElementById('currentDate');
    const now = new Date();
    const options = { weekday: 'long', day: 'numeric', month: 'long' };
    const formattedDate = now.toLocaleDateString('en-US', options);

    // Format: "Friday, 10th February"
    const day = now.getDate();
    const suffix = getDaySuffix(day);
    const monthYear = now.toLocaleDateString('en-US', { month: 'long' });
    const weekday = now.toLocaleDateString('en-US', { weekday: 'long' });

    dateElement.textContent = `${weekday}, ${day}${suffix} ${monthYear}`;
}

function getDaySuffix(day) {
    if (day > 3 && day < 21) return 'th';
    switch (day % 10) {
        case 1: return 'st';
        case 2: return 'nd';
        case 3: return 'rd';
        default: return 'th';
    }
}

// Setup event listeners
function setupEventListeners() {
    // File upload handler
    const fileInput = document.getElementById('fileUpload');
    if (fileInput) fileInput.addEventListener('change', handleFileUpload);

    // View details buttons
    const viewDetailsBtns = document.querySelectorAll('.view-details-btn');
    viewDetailsBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            const card = e.target.closest('.period-card');
            const period = card.dataset.period;

            let periodKey = '';
            if (period === '1to7') periodKey = '1-7_days';
            else if (period === 'more7') periodKey = 'More_than_7_days';
            else if (period === 'today') periodKey = 'Today';

            showSummaryDetailsListView(periodKey);
        });
    });

    // Back button
    const backBtn = document.getElementById('backToDashboard');
    if (backBtn) {
        backBtn.addEventListener('click', () => {
            showView('dashboard');
        });
    }

    // Make bulk call button
    const makeBulkCallBtn = document.getElementById('makeBulkCallBtn');
    if (makeBulkCallBtn) {
        makeBulkCallBtn.addEventListener('click', handleBulkCall);
    }

    // Modal close
    const closeBtn = document.querySelector('.close-btn');
    const modal = document.getElementById('detailsModal');
    if (closeBtn) {
        closeBtn.addEventListener('click', () => {
            modal.classList.remove('active');
        });
    }

    if (modal) {
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.classList.remove('active');
            }
        });
    }

    // Sidebar navigation
    const navItems = document.querySelectorAll('.nav-item');
    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const targetView = item.getAttribute('data-view');
            showView(targetView);
        });
    });
}

// Helper to switch views
function showView(viewId) {
    const sections = document.querySelectorAll('.view-section');
    const navItems = document.querySelectorAll('.nav-item');
    const headerActions = document.getElementById('headerActions');

    // Reset state
    currentView = viewId;
    sessionStorage.setItem('current_view', viewId);

    // Allow upload in Dashboard and Reports
    if (viewId === 'dashboard' || viewId === 'reports') {
        if (headerActions) headerActions.style.display = 'flex';
    } else {
        if (headerActions) headerActions.style.display = 'none';
    }

    if (viewId === 'dashboard') {
        currentBorrowerId = null;
        sessionStorage.removeItem('current_borrower_id');
        sessionStorage.removeItem('current_period_key');
    }

    // Update Nav
    navItems.forEach(nav => {
        if (nav.getAttribute('data-view') === viewId) {
            nav.classList.add('active');
        } else {
            nav.classList.remove('active');
        }
    });

    // Update Sections
    sections.forEach(section => {
        section.classList.remove('active');
    });

    const targetElement = document.getElementById(`${viewId}-view`);
    if (targetElement) {
        targetElement.classList.add('active');
        if (viewId === 'reports') {
            fetchReportData();
        }
    }
}

// Fetch dashboard data from server (sync with cache)
async function fetchDashboardData() {
    console.log('Fetching dashboard data from server...');
    try {
        // POST to /data without file returns cached data
        // Using POST because unified_data_endpoint is POST
        const response = await fetch(`${API_BASE_URL}/data_ingestion/data?include_details=true`, {
            method: 'POST'
        });

        if (response.ok) {
            const data = await response.json();
            if (data.status === 'success' && data.kpis) {
                console.log('✅ Dashboard data fetched from server cache');
                currentKpiData = data;
                localStorage.setItem('finance_data', JSON.stringify(data));
                updateDashboard(data);
            } else {
                console.log('ℹ️ No cached dashboard data available on server');
            }
        } else {
            if (response.status === 400) {
                console.log('ℹ️ Server cache is empty');
            } else {
                console.error('❌ Error fetching dashboard data:', response.status);
            }
        }
    } catch (error) {
        console.error('❌ Network error fetching dashboard data:', error);
    }
}

// Handle file upload
async function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    console.log('File upload started:', file.name);

    // Validate file type
    const validExtensions = ['.xlsx', '.xls', '.csv'];
    const fileName = file.name.toLowerCase();
    const isValid = validExtensions.some(ext => fileName.endsWith(ext));

    if (!isValid) {
        alert('Please upload a valid Excel or CSV file (.xlsx, .xls, .csv)');
        event.target.value = '';
        return;
    }

    showLoading(true);

    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${API_BASE_URL}/data_ingestion/data?include_details=true`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to upload file');
        }

        const data = await response.json();
        console.log('API Response received successfully');

        // Reset call states for new data
        if (data.detailed_breakdown?.by_due_date_category) {
            Object.values(data.detailed_breakdown.by_due_date_category).flat().forEach(b => {
                b.call_in_progress = false;
                b.call_completed = false;
            });
        }

        currentKpiData = data;
        // Persist data so it survives reloads
        localStorage.setItem('finance_data', JSON.stringify(data));
        console.log('✅ Data persisted to localStorage');

        updateDashboard(data);
        showNotification('File uploaded successfully!', 'success');
    } catch (error) {
        console.error('Upload error:', error);
        showNotification(`Error: ${error.message}`, 'error');
    } finally {
        showLoading(false);
        event.target.value = ''; // Reset file input
    }
}

// Update dashboard with KPI data
function updateDashboard(data) {
    if (!data || !data.kpis) return;

    // Update overview KPIs
    const borrowersEl = document.getElementById('totalBorrowers');
    const arrearsEl = document.getElementById('totalArrears');

    if (borrowersEl) borrowersEl.textContent = data.kpis.total_borrowers || 0;
    if (arrearsEl) arrearsEl.textContent =
        `₹${(data.kpis.total_arrears || 0).toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

    if (data.detailed_breakdown && data.detailed_breakdown.by_due_date_category) {
        const byDate = data.detailed_breakdown.by_due_date_category;
        updateCardLocal('more7', byDate['More_than_7_days']);
        updateCardLocal('oneToSeven', byDate['1-7_days']);
        updateCardLocal('today', byDate['Today']);
    }
}

// Helper to calculate counts locally and update UI
function updateCardLocal(prefix, borrowersList) {
    if (!borrowersList || !Array.isArray(borrowersList)) {
        document.querySelector(`#${prefix}-consistent .count`).textContent = 0;
        document.querySelector(`#${prefix}-inconsistent .count`).textContent = 0;
        document.querySelector(`#${prefix}-overdue .count`).textContent = 0;
        return;
    }

    let consistent = 0, inconsistent = 0, overdue = 0;

    borrowersList.forEach(b => {
        const category = b.Payment_Category;
        if (category === 'Consistent') consistent++;
        else if (category === 'Inconsistent') inconsistent++;
        else if (category === 'Overdue') overdue++;
    });

    document.querySelector(`#${prefix}-consistent .count`).textContent = consistent;
    document.querySelector(`#${prefix}-inconsistent .count`).textContent = inconsistent;
    document.querySelector(`#${prefix}-overdue .count`).textContent = overdue;
}

// Show Summary Details List View
function showSummaryDetailsListView(periodKey) {
    console.log('Showing summary details list for period:', periodKey);

    if (!currentKpiData || !currentKpiData.detailed_breakdown) {
        showNotification('No data available. Please upload a file.', 'warning');
        return;
    }

    const byDate = currentKpiData.detailed_breakdown.by_due_date_category;
    const borrowers = byDate[periodKey] || [];

    // Map keys to labels
    const periodLabels = {
        'More_than_7_days': 'More than 7 Days',
        '1-7_days': '1-7 Days',
        'Today': '6th Feb (Today Data)'
    };

    const labelEl = document.getElementById('selectedPeriodLabel');
    if (labelEl) labelEl.textContent = periodLabels[periodKey] || periodKey;

    // Reset any stale call states for these borrowers when opening the view fresh
    borrowers.forEach(b => {
        if (!b.call_completed) { // Only reset if not already successful
            b.call_in_progress = false;
        }
    });

    // Save state
    currentView = 'summary-details';
    sessionStorage.setItem('current_view', currentView);
    sessionStorage.setItem('current_period_key', periodKey);

    // Switch view
    showView('summary-details');

    // Populate rows
    const container = document.getElementById('callRowsContainer');
    container.innerHTML = '';

    if (borrowers.length === 0) {
        container.innerHTML = '<div style="text-align: center; padding: 40px; color: #6b7280;">No borrowers found in this section.</div>';
        return;
    }

    borrowers.forEach(borrower => {
        const rowWrapper = createCallDataRow(borrower);
        container.appendChild(rowWrapper);
    });

    window.scrollTo(0, 0);
}

// Create a call data row
function createCallDataRow(borrower) {
    const wrapper = document.createElement('div');
    wrapper.className = 'call-row-wrapper';
    wrapper.id = `row-${borrower.NO}`;

    const interactionType = borrower.Payment_Category || 'Normal';
    const statusClass = interactionType.toLowerCase();

    // Call Status Logic
    let callStatus = "Yet To Call";
    let statusBtnClass = "yet-to-call";

    if (borrower.call_in_progress) {
        callStatus = "In progress";
        statusBtnClass = "in-progress";
    } else if (borrower.call_completed) {
        callStatus = "Call Success";
        statusBtnClass = "success";
    }

    const lastPaid = borrower.LAST_PAID_DATE || borrower.DUE_DATE || 'N/A';
    const amount = (borrower.AMOUNT || 0).toLocaleString('en-IN', { minimumFractionDigits: 2 });
    const totalAmount = (borrower.TOTAL_LOAN || (borrower.AMOUNT * 1.5) || 0).toLocaleString('en-IN', { minimumFractionDigits: 2 });

    wrapper.innerHTML = `
        <div class="call-row">
            <div class="borrower-cell">
                <img src="https://ui-avatars.com/api/?name=${encodeURIComponent(borrower.BORROWER)}&background=random" class="borrower-avatar" alt="${borrower.BORROWER}">
                <div class="borrower-meta">
                    <h4>${borrower.BORROWER}</h4>
                    <p>Last paid: ${lastPaid}</p>
                </div>
            </div>
            <div class="due-cell">$${amount}</div>
            <div class="total-cell">$${totalAmount}</div>
            <div class="status-cell ${statusClass}">${interactionType}</div>
            <div class="action-cell">
                <button class="status-btn ${statusBtnClass}">
                    <span>${callStatus}</span>
                    <span class="dropdown-icon">▼</span>
                </button>
            </div>
        </div>
        <div class="expanded-content">
            <div class="conversation-card">
                <div class="card-header">
                    <span class="icon">✨</span> AI Conversation
                </div>
                <div class="chat-bubbles" id="transcript-${borrower.NO}">
                    ${renderTranscript(borrower.transcript)}
                </div>
            </div>
            <div class="summary-card" id="summary-card-${borrower.NO}">
                <div class="card-header">
                    <span class="icon">✨</span> AI Summary
                </div>
                <div class="next-steps-title">Next Steps</div>
                <div class="next-steps-text" id="summary-text-${borrower.NO}">
                    ${borrower.ai_summary || 'No call summary yet. Initiate a call to get AI insights.'}
                </div>
                <button class="manual-btn">Initiate Manual Process</button>
            </div>
        </div>
    `;

    // Toggle expansion
    wrapper.querySelector('.call-row').addEventListener('click', () => {
        wrapper.classList.toggle('expanded');
    });

    return wrapper;
}

// Render transcript bubbles
function renderTranscript(transcript) {
    if (!transcript || transcript.length === 0) {
        return '<div class="chat-bubble ai">No conversation recorded yet.</div>';
    }

    return transcript.map(t => `
        <div class="chat-bubble ${t.speaker.toLowerCase() === 'ai' ? 'ai' : 'person'}">
            ${t.text}
        </div>
    `).join('');
}

// Handle bulk call
async function handleBulkCall() {
    const periodKey = sessionStorage.getItem('current_period_key');
    if (!periodKey || !currentKpiData) return;

    const borrowers = currentKpiData.detailed_breakdown.by_due_date_category[periodKey] || [];
    if (borrowers.length === 0) {
        showNotification('No borrowers to call.', 'warning');
        return;
    }

    showNotification(`Triggering parallel calls for ${borrowers.length} borrowers...`, 'info');

    const makeBulkCallBtn = document.getElementById('makeBulkCallBtn');
    if (makeBulkCallBtn) makeBulkCallBtn.disabled = true;

    // Update UI to "In progress"
    borrowers.forEach(b => {
        b.call_in_progress = true;
        b.call_completed = false;

        const row = document.getElementById(`row-${b.NO}`);
        if (row) {
            const btn = row.querySelector('.status-btn');
            if (btn) {
                btn.className = 'status-btn in-progress';
                const span = btn.querySelector('span');
                if (span) span.textContent = 'In progress';
            }
        }
    });

    try {
        const payload = {
            borrowers: borrowers.map(b => ({
                NO: String(b.NO || ''),
                cell1: String(b.cell1 || ''),
                preferred_language: String(b.preferred_language || 'en-IN')
            })),
            use_dummy_data: true
        };

        const response = await fetch(`${API_BASE_URL}/ai_calling/trigger_calls`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) throw new Error('Bulk call request failed');

        const result = await response.json();
        console.log('Bulk Call Results:', result);

        // Update local state and UI
        result.results.forEach(res => {
            // Use loose equality (==) to handle string vs number comparison
            const borrower = borrowers.find(b => b.NO == res.borrower_id);
            if (borrower) {
                console.log(`Updating UI for borrower ${res.borrower_id}`);
                borrower.call_in_progress = false;
                borrower.call_completed = res.success;
                borrower.ai_summary = res.ai_analysis ? res.ai_analysis.summary : (res.success ? 'Call completed.' : 'Call failed: ' + res.error);
                borrower.transcript = res.conversation || [];

                // Update Row UI
                const row = document.getElementById(`row-${borrower.NO}`);
                if (row) {
                    const btn = row.querySelector('.status-btn');
                    if (btn) {
                        const span = btn.querySelector('span');
                        if (res.success) {
                            btn.className = 'status-btn success';
                            if (span) span.textContent = 'Call Success';
                        } else {
                            btn.className = 'status-btn yet-to-call';
                            if (span) span.textContent = 'Yet To Call';
                        }
                    }

                    // Update Transcript in expanded content
                    const transcriptEl = document.getElementById(`transcript-${borrower.NO}`);
                    if (transcriptEl) {
                        transcriptEl.innerHTML = renderTranscript(borrower.transcript);
                    }

                    // Update Summary in expanded content
                    const summaryEl = document.getElementById(`summary-text-${borrower.NO}`);
                    if (summaryEl) {
                        summaryEl.textContent = borrower.ai_summary;
                    }
                }
            } else {
                console.warn(`Could not find borrower ${res.borrower_id} in current list to update UI.`);
            }
        });

        // Save state
        localStorage.setItem('finance_data', JSON.stringify(currentKpiData));
        showNotification(`Bulk call completed! ${result.successful_calls} successful.`, 'success');

    } catch (error) {
        console.error('Bulk call error:', error);
        showNotification(`Error: ${error.message}`, 'error');

        // Reset progress status on error
        borrowers.forEach(b => {
            b.call_in_progress = false;
            const row = document.getElementById(`row-${b.NO}`);
            if (row) {
                const btn = row.querySelector('.status-btn');
                if (btn) {
                    btn.className = 'status-btn yet-to-call';
                    btn.querySelector('span').textContent = 'Yet To Call';
                }
            }
        });
    } finally {
        if (makeBulkCallBtn) makeBulkCallBtn.disabled = false;
    }
}

// Show/hide loading spinner
function showLoading(show) {
    const spinner = document.getElementById('loadingSpinner');
    spinner.style.display = show ? 'flex' : 'none';
}

// Show notification (basic version)
function showNotification(message, type = 'info') {
    // You can enhance this with a proper toast notification library
    const styles = {
        success: 'background: #10b981; color: white;',
        error: 'background: #ef4444; color: white;',
        warning: 'background: #f59e0b; color: white;',
        info: 'background: #3b82f6; color: white;'
    };

    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 16px 24px;
        border-radius: 12px;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        z-index: 3000;
        animation: slideInRight 0.3s ease;
        ${styles[type] || styles.info}
    `;
    notification.textContent = message;

    document.body.appendChild(notification);

    setTimeout(() => {
        notification.style.animation = 'slideOutRight 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// Add animation styles
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    @keyframes slideOutRight {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(400px);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);


// ==========================================
// REPORT DATA HANDLING
// ==========================================

async function fetchReportData() {
    const tableBody = document.getElementById('reportsTableBody');
    if (!tableBody) return;

    tableBody.innerHTML = '<tr><td colspan="9" style="text-align: center; padding: 20px;">Loading report data...</td></tr>';

    try {
        const response = await fetch(`${API_BASE_URL}/data_ingestion/report_data`);
        const result = await response.json();

        if (result.status === 'success') {
            renderReportTable(result.data);
        } else {
            tableBody.innerHTML = `<tr><td colspan="9" style="text-align: center; padding: 20px; color: #ef4444;">${result.message}</td></tr>`;
        }
    } catch (error) {
        console.error('Failed to fetch report data:', error);
        tableBody.innerHTML = '<tr><td colspan="9" style="text-align: center; padding: 20px; color: #ef4444;">Failed to load data. Please try again.</td></tr>';
    }
}

function renderReportTable(data) {
    const tableBody = document.getElementById('reportsTableBody');
    if (!tableBody) return;

    if (!data || data.length === 0) {
        tableBody.innerHTML = '<tr><td colspan="9" style="text-align: center; padding: 20px;">No data available.</td></tr>';
        return;
    }

    tableBody.innerHTML = data.map((row, index) => `
        <tr style="border-bottom: 1px solid rgba(255,255,255,0.05); hover: {background: rgba(255,255,255,0.05)}">
            <td style="padding: 12px 16px;">${row.NO}</td>
            <td style="padding: 12px 16px;">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <img src="https://ui-avatars.com/api/?name=${encodeURIComponent(row.BORROWER)}&background=random&size=32" style="width: 32px; height: 32px; border-radius: 50%; object-fit: cover;" alt="${row.BORROWER}">
                    <span style="font-weight: 500;">${row.BORROWER}</span>
                </div>
            </td>
            <td style="padding: 12px 16px;">${row.AMOUNT}</td>
            <td style="padding: 12px 16px;">${row.cell1}</td>
            <td style="padding: 12px 16px;">${row.EMI}</td>
            <td style="padding: 12px 16px;">${row.LAST_DUE_REVD_DATE}</td>
            <td style="padding: 12px 16px;">${row.FIRST_DUE_DATE}</td>
            <td style="padding: 12px 16px;">${row.preferred_language}</td>
            <td style="padding: 12px 16px;">${row.PAYMENT_CONFIRMATION}</td>
            <td style="padding: 12px 16px;">${row.DATE}</td>
            <td style="padding: 12px 16px;">
                <span style="
                    padding: 4px 8px; 
                    border-radius: 4px; 
                    font-size: 12px; 
                    background: ${getStatusColor(row.CALL_STATUS)};
                    color: white;
                ">
                    ${row.CALL_STATUS || 'Pending'}
                </span>
            </td>
        </tr>
    `).join('');
}

function getStatusColor(status) {
    if (!status) return '#6b7280'; // Gray
    status = status.toLowerCase();
    if (status.includes('success') || status === 'completed') return '#10b981'; // Green
    if (status.includes('fail') || status === 'busy') return '#ef4444'; // Red
    if (status.includes('progress')) return '#f59e0b'; // Orange
    return '#3b82f6'; // Blue
}

function exportTableToCSV(filename) {
    const csv = [];
    const rows = document.querySelectorAll("#reportsTable tr");

    for (let i = 0; i < rows.length; i++) {
        const row = [], cols = rows[i].querySelectorAll("td, th");

        for (let j = 0; j < cols.length; j++)
            row.push('"' + cols[j].innerText + '"');

        csv.push(row.join(","));
    }

    const csvFile = new Blob([csv.join("\n")], { type: "text/csv" });
    const downloadLink = document.createElement("a");
    downloadLink.download = filename;
    downloadLink.href = window.URL.createObjectURL(csvFile);
    downloadLink.style.display = "none";
    document.body.appendChild(downloadLink);
    downloadLink.click();
}

// Attach global function for CSV export
window.exportTableToCSV = exportTableToCSV;

// Add refresh button listener
const refreshBtn = document.getElementById('refreshReportsBtn');
if (refreshBtn) {
    refreshBtn.addEventListener('click', fetchReportData);
}


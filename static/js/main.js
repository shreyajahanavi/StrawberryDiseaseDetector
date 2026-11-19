/**
 * BerryGuard AI - Main Global JavaScript
 * Handles navigation drawer, smooth interactions, and global utilities.
 */

document.addEventListener('DOMContentLoaded', () => {
    // Mobile Drawer Navigation
    const mobileToggle = document.getElementById('mobileToggle');
    const mobileDrawer = document.getElementById('mobileDrawer');
    const drawerClose = document.getElementById('drawerClose');
    const drawerBackdrop = document.getElementById('drawerBackdrop');

    function openDrawer() {
        if (mobileDrawer) mobileDrawer.classList.add('open');
        if (drawerBackdrop) drawerBackdrop.classList.add('active');
        document.body.style.overflow = 'hidden';
    }

    function closeDrawer() {
        if (mobileDrawer) mobileDrawer.classList.remove('open');
        if (drawerBackdrop) drawerBackdrop.classList.remove('active');
        document.body.style.overflow = '';
    }

    if (mobileToggle) mobileToggle.addEventListener('click', openDrawer);
    if (drawerClose) drawerClose.addEventListener('click', closeDrawer);
    if (drawerBackdrop) drawerBackdrop.addEventListener('click', closeDrawer);

    // Close drawer on escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeDrawer();
    });

    // Check Backend Model Status on Page Load
    fetch('/api/model-status')
        .then(res => res.json())
        .then(data => {
            if (data && data.status) {
                console.log('🍓 BerryGuard AI Backend Status:', data.status.status_message);
            }
        })
        .catch(err => {
            console.warn('Backend status check unreachable:', err);
        });
});

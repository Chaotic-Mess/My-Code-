// Adds hover animation for the bookmarklet button
document.getElementById("bookmarklet-btn").addEventListener("click", e => {
  alert("To use this tool, drag the button to your bookmarks bar and click it while watching a YouTube video!");
  e.preventDefault();
});

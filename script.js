async function checkSpam() {
    const text = document.getElementById("emailText").value;

    try {
        const response = await fetch("http://127.0.0.1:5000/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ text: text })
        });

        const data = await response.json();

        document.getElementById("result").innerHTML =
            `Result: ${data.result} <br> Probability: ${Number(data.probability).toFixed(2)}`;
    } catch (error) {
        document.getElementById("result").innerText = "Error connecting to server";
    }
}
fetch('models').then(response=>{
    console.log(response)
    return response.json()
}).then(entry=>{
    console.log(entry)
    for (ent of entry.models){
        document.getElementById("table").innerHTML += "<tr><th><a onclick='testing(this)'>"+ent.id+"</a></th><td>"+ent.date_time+"</td><td>"+ent.cv_score+"</td><td>"+ent.features+"</td></tr>"
    }
});

document.getElementById("training").addEventListener("click", function(e){
    e.preventDefault();
    fetch('model?data_path=..%2Ftrain_iris.csv', {method:"POST"}).then(response=>{
        return response.json();
    }).then(entry=>{
        console.log(entry)
        if(!alert(entry.model_id+' model is saved.')){window.location.reload();}
    })
});

function testing(element){
    fetch('model?data_path=..%2Ftest_iris.csv&model_id='+element.innerHTML).then(response=>{
        return response.json();
    }).then(entry=>{
        alert('testing acc is '+entry.acc);
    })
}

localurl = 'http://localhost:5000/predict'
remoteurl = 'https://mammass.herokuapp.com/predict'
let success = (data) => {
  console.log(data)
}
let $loading = $('#loading').hide();
$().ready(() => {
  console.log("Hello world")
  $("#malignant").hide()
  $("#benign").hide()
  $( "#predict" ).submit(function( event ) {
    console.log("Form submitted")
    event.preventDefault();
    let $form = $(this)
    let birads = parseFloat($form.find("select[name='birads']").val())
    let age = parseFloat($form.find("input[name='age']").val())
    let shape = parseFloat($form.find("select[name='shape']").val())
    let margin = parseFloat($form.find("select[name='margin']").val())
    let density = parseFloat($form.find("select[name='density']").val())
    features = [birads, age, shape, margin, density]
    data = { 'features': [features] }
    console.log(features)
    $loading.show();
    $("#malignant").hide()
    $("#benign").hide()
    $.ajax({
      type: "POST",
      url: remoteurl,
      data: JSON.stringify(data),
      contentType: 'application/json'
    })
      .done((data) => {
        $loading.hide();
        parsed = JSON.parse(data)
        probs = parsed['probabilities']
        let benign = probs[0][0]*100
        benign = benign.toFixed(2)
        let malignant = probs[0][1]*100
        malignant = malignant.toFixed(2)
        console.log(benign)
        console.log(malignant)
        if (benign > malignant) {
          $("#malignant").hide()
          $("#percent1").html(benign)
          $("#benign").show()
        } else {
          $("#benign").hide()
          $("#percent2").html(malignant)
          $("#malignant").show()
        }
      })
      .fail((xhr) => {
        console.log(xhr.status)
      })
    console.log('data posted')
  })
})


    var trace1 = {
      name: 'Energy consumption',
      x: ['1-3-2019', '2-3-2019', '3-3-2019', '4-3-2019', '5-3-2019', '6-3-2019', '7-3-2019'],
      y: [10, 15, 13, 17, 19, 10, 11, 13],
      type: 'line',
      line: {
        color: '#45B5C6'
      }
    };

    var trace2 = {
      name: 'Cold water consumption',
      x: ['1-3-2019', '2-3-2019', '3-3-2019', '4-3-2019', '5-3-2019', '6-3-2019', '7-3-2019'],
      y: [16, 5, 11, 9, 0, 4, 0],
      type: 'line'
    };

    var data = [trace1, trace2];

    layout = {
        showlegend: true
    }

    Plotly.newPlot('div-id', data, layout, {
      responsive: true,
      displaylogo: false
    });


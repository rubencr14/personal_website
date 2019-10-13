import {Boston} from "/machinelearn/datasets";
alert("PENE")
async function dat(){

    const bostonData = new Boston();
    const {

        data,
        targets,
        label

    } = await bostonData.load();
    document.write(data)

};

dat()
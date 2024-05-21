const { Firestore } = require('@google-cloud/firestore');

async function storeData(id, data) {
    try {
        const db = new Firestore();
        const predictCollection = db.collection('prediction');
        await predictCollection.doc(id).set(data);
        console.log('Data stored successfully');
    } catch (error) {
        console.error('Error storing data:', error);
        throw new Error('Failed to store data');
    }
}

module.exports = storeData;


/***************/
/* Misc. Utils */
/***************/

const isUndefined = value => value === void(0);

const numberToOrdinal = (number) => {
    const onesDigit = number % 10;
    const tensDigit = number % 100;
    if (onesDigit == 1 && tensDigit != 11) {
        return number + 'st';
    } else if (onesDigit == 2 && tensDigit != 12) {
        return number + 'nd';
    } else if (onesDigit == 3 && tensDigit != 13) {
        return number + 'rd';
    } else {
        return number + 'th';
    }
};

const numberToCommaSeparatedNumberString = number => number.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");

/**********************/
/* HTML Element Utils */
/**********************/

const removeAllChildNodes = (parent) => {
    while (parent.firstChild) {
        parent.removeChild(parent.firstChild);
    }
};

const createNewElement = (childTag, {classes, attributes, innerHTML}={}) => {
    const newElement = childTag === 'svg' ? document.createElementNS('http://www.w3.org/2000/svg', childTag) : document.createElement(childTag);
    if (!isUndefined(classes)) {
        classes.forEach(childClass => newElement.classList.add(childClass));
    }
    if (!isUndefined(attributes)) {
        Object.entries(attributes).forEach(([attributeName, attributeValue]) => {
            newElement.setAttribute(attributeName, attributeValue);
        });
    }
    if (!isUndefined(innerHTML)) {
        newElement.innerHTML = innerHTML;
    }
    return newElement;
};

const createTableWithElements = (rows, {classes, attributes}={}) => {
    const table = createNewElement('table', {classes, attributes});
    rows.forEach(elements => {
        const tr = document.createElement('tr');
        table.append(tr);
        elements.forEach(element => {
            const td = document.createElement('td');
            tr.append(td);
            td.append(element);
        });
    });
    return table;
};

const createLazyAccordion = (labelGeneratorDestructorTriples) => {
    const accordionContainer = createNewElement('div', {classes: ['accordion-container']});
    labelGeneratorDestructorTriples.forEach(([labelInnerHTML, contentGenerator, contentDestructor], i) => {
        // contentGenerator and contentDestructor take an HTML element
        const labelContentPairElement = createNewElement('div');
        const label = createNewElement('p', {classes: ['accordion-label'], innerHTML: labelInnerHTML});
        const contentDiv = createNewElement('div', {classes: ['accordion-content']});
        labelContentPairElement.append(label);
        labelContentPairElement.append(contentDiv);
        label.onclick = () => {
            label.classList.toggle('active');
            contentDiv.classList.toggle('active');
            if (label.classList.contains('active')) {
                contentGenerator(contentDiv);
            } else {
                contentDestructor(contentDiv);
            }
        };
        accordionContainer.append(labelContentPairElement);
    });
    return accordionContainer;
};

/********/
/* Main */
/********/

const numberOfBestTrialsToDisplay = 25;

const generateLabelGeneratorDestructorTriples = (resultDicts) => 
      resultDicts.sort((resultSummaryDict1, resultSummaryDict2) => resultSummaryDict1.best_validation_loss-resultSummaryDict2.best_validation_loss).slice(0, numberOfBestTrialsToDisplay).map(({
          // Results
          best_validation_loss,
          testing_accuracy,
          testing_correctness_count,
          testing_loss,
          // Hyperparameters
          embedding_size,
          p,
          q,
          walk_length,
          walks_per_node,
          node2vec_epochs,
          node2vec_learning_rate,
          link_predictor_learning_rate,
          link_predictor_batch_size,
          link_predictor_gradient_clip_val,
          // Misc.
          best_validation_model_path,
          duration_seconds,
          testing_portion,
          testing_set_size,
          training_portion,
          training_set_size,
          validation_portion,
          validation_set_size,
          trial_index,
      }, rank_index) => {
          const labelInnerHTML = `${numberToOrdinal(rank_index+1)} Place (Trial #${trial_index})`;
          const contentGenerator = (contentContainer) => {
              const hyperparameterTableContainer = createNewElement('div', {classes: ['result-table-container']});
              contentContainer.append(hyperparameterTableContainer);
              removeAllChildNodes(hyperparameterTableContainer);
              hyperparameterTableContainer.append(createNewElement('p', {innerHTML: 'Results', attributes: {style: 'margin: 2em 0px 10px 0px'}}));
              hyperparameterTableContainer.append(
                  createTableWithElements([
                      [createNewElement('p', {innerHTML: `Validation Loss:`}), createNewElement('p', {attributes: {style: 'float: right;'}, innerHTML: `${best_validation_loss}`})],
                      [createNewElement('p', {innerHTML: `Testing Loss:`}), createNewElement('p', {attributes: {style: 'float: right;'}, innerHTML: `${testing_loss}`})],
                      [
                          createNewElement('p', {innerHTML: `Testing Accuracy:`}),
                          createNewElement('p', {
                              attributes: {style: 'float: right;'},
                              innerHTML: `${numberToCommaSeparatedNumberString(testing_correctness_count)}/${numberToCommaSeparatedNumberString(testing_set_size)} (${(testing_accuracy*100).toFixed(2)}%)`
                          })
                      ],
                  ], {classes: ['result-table'], attributes: {style: 'margin-bottom: 1em;'}})
              );
              hyperparameterTableContainer.append(createNewElement('p', {innerHTML: 'Hyperparameters', attributes: {style: 'margin: 2em 0px 10px 0px'}}));
              hyperparameterTableContainer.append(
                  createTableWithElements([
                      [createNewElement('p', {innerHTML: `node2vec Embedding Size:`}), createNewElement('p', {attributes: {style: 'float: right;'}, innerHTML: `${embedding_size}`})],
                      [createNewElement('p', {innerHTML: `node2vec Random Walk Parameter <i>p</i>:`}), createNewElement('p', {attributes: {style: 'float: right;'}, innerHTML: `${p}`})],
                      [createNewElement('p', {innerHTML: `node2vec Random Walk Parameter <i>q</i>:`}), createNewElement('p', {attributes: {style: 'float: right;'}, innerHTML: `${q}`})],
                      [createNewElement('p', {innerHTML: `node2vec Random Walk Length:`}), createNewElement('p', {attributes: {style: 'float: right;'}, innerHTML: `${walk_length}`})],
                      [createNewElement('p', {innerHTML: `node2vec Number of Random Walks Per Node:`}), createNewElement('p', {attributes: {style: 'float: right;'}, innerHTML: `${walks_per_node}`})],
                      [createNewElement('p', {innerHTML: `node2vec Number of Epochs:`}), createNewElement('p', {attributes: {style: 'float: right;'}, innerHTML: `${node2vec_epochs}`})],
                      [createNewElement('p', {innerHTML: `node2vec Learning Rate:`}), createNewElement('p', {attributes: {style: 'float: right;'}, innerHTML: `${node2vec_learning_rate}`})],
                      [createNewElement('p', {innerHTML: `Logistic Regression Initial Learning Rate:`}), createNewElement('p', {attributes: {style: 'float: right;'}, innerHTML: `${link_predictor_learning_rate}`})],
                      [createNewElement('p', {innerHTML: `Logistic Regression Batch Size:`}), createNewElement('p', {attributes: {style: 'float: right;'}, innerHTML: `${link_predictor_batch_size}`})],
                      [
                          createNewElement('p', {innerHTML: `Logistic Regression Training Max Gradient Clipping Threshold:`}),
                          createNewElement('p', {attributes: {style: 'float: right;'}, innerHTML: `${link_predictor_gradient_clip_val}`})
                      ],
                  ], {classes: ['result-table'], attributes: {style: 'margin-bottom: 1em;'}})
              );
              hyperparameterTableContainer.append(createNewElement('p', {innerHTML: 'Training Time', attributes: {style: 'margin: 2em 0px 10px 0px'}}));
              const numberOfCompletedEpochs = parseInt(best_validation_model_path.split('/').pop().split('.')[0].split('=')[1].split('_')[0])+1;
              const numberOfTrainingSteps = Math.floor(training_set_size/link_predictor_batch_size) * numberOfCompletedEpochs;
              hyperparameterTableContainer.append(
                  createTableWithElements([
                      [
                          createNewElement('p', {innerHTML: `Total Training Time:`}),
                          createNewElement('p', {attributes: {style: 'float: right;'}, innerHTML: `${numberToCommaSeparatedNumberString(duration_seconds)} seconds`})
                      ],
                      [
                          createNewElement('p', {innerHTML: `Number of Logistic Regression Epochs Until Early Stop:`}),
                          createNewElement('p', {attributes: {style: 'float: right;'}, innerHTML: `${numberToCommaSeparatedNumberString(numberOfCompletedEpochs)}`})
                      ],
                      [
                          createNewElement('p', {innerHTML: `Number of Logistic Regression Training Steps Until Early Stop:`}),
                          createNewElement('p', {attributes: {style: 'float: right;'}, innerHTML: `${numberToCommaSeparatedNumberString(numberOfTrainingSteps)}`})
                      ],
                  ], {classes: ['result-table'], attributes: {style: 'margin-bottom: 1em;'}})
              );
              hyperparameterTableContainer.append(createNewElement('p', {innerHTML: 'Dataset Details', attributes: {style: 'margin: 2em 0px 10px 0px'}}));
              hyperparameterTableContainer.append(
                  createTableWithElements([
                      [createNewElement('p', {innerHTML: `Logistic Regression Training Set Portion:`}), createNewElement('p', {attributes: {style: 'float: right;'}, innerHTML: `${100*training_portion}%`})],
                      [createNewElement('p', {innerHTML: `Logistic Regression Validation Set Portion:`}), createNewElement('p', {attributes: {style: 'float: right;'}, innerHTML: `${100*validation_portion}%`})],
                      [createNewElement('p', {innerHTML: `Logistic Regression Testing Set Portion:`}), createNewElement('p', {attributes: {style: 'float: right;'}, innerHTML: `${100*testing_portion}%`})],
                      [
                          createNewElement('p', {innerHTML: `Logistic Regression Training Set Size:`}),
                          createNewElement('p', {attributes: {style: 'float: right;'}, innerHTML: `${numberToCommaSeparatedNumberString(training_set_size)}`})
                      ],
                      [
                          createNewElement('p', {innerHTML: `Logistic Regression Validation Set Size:`}),
                          createNewElement('p', {attributes: {style: 'float: right;'}, innerHTML: `${numberToCommaSeparatedNumberString(validation_set_size)}`})
                      ],
                      [
                          createNewElement('p', {innerHTML: `Logistic Regression Testing Set Size:`}),
                          createNewElement('p', {attributes: {style: 'float: right;'}, innerHTML: `${numberToCommaSeparatedNumberString(testing_set_size)}`})
                      ],
                  ], {classes: ['result-table'], attributes: {style: 'margin-bottom: 1em;'}})
              );
          };
          const contentDestructor = removeAllChildNodes;

          return [labelInnerHTML, contentGenerator, contentDestructor];
      });

d3.json('./hyperparameter_search_results.json')
    .then(resultDicts => {
        document.querySelector('#number-of-hyperparameter-trials-span').innerHTML = `${resultDicts.length}`;
        document.querySelector('#number-of-best-trials-span').innerHTML = `${numberOfBestTrialsToDisplay}`;
        const resultDiv = document.querySelector('#hyperparameter-results');
        const labelGeneratorDestructorTriples = generateLabelGeneratorDestructorTriples(resultDicts);
        const accordion = createLazyAccordion(labelGeneratorDestructorTriples);
        removeAllChildNodes(resultDiv);
        resultDiv.append(accordion);
    }).catch(err => {
        console.error(err.message);
        return;
    });

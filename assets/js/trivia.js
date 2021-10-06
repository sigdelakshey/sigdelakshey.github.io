var triviaBtns = document.querySelectorAll('.trivia-button');
var body = document.querySelector('body');

var openedPopOver = null;

function hidePopOver(popOverID) {
    popOver = document.getElementById(popOverID);
    popOver.classList.remove('active');
    popOver.classList.add('inactive');
    openedPopOver = null;
}

function showPopOver(event) {
    targetID = event.target.getAttribute('id');
    identifier = targetID.match(/trivia-id-(.*)/)[1];
    popOverID = 'trivia-content-' + identifier;
    
    if (openedPopOver != null & openedPopOver != popOverID){
        // some other popOver is already open. So first close that.
        hidePopOver(openedPopOver);
    }

    //Set the opened popOver ID
    openedPopOver = popOverID;
    // console.log('Showing: ', openedPopOver);

    popOver = document.getElementById(popOverID);
    popOver.classList.remove('inactive');
    popOver.classList.add('active');

    box_rect = popOver.parentElement.getBoundingClientRect();
    rect = popOver.getBoundingClientRect();

    xPos = event.clientX - box_rect['x'];
    yPos = event.clientY - box_rect['y'];

    popCoverage = rect['width']/box_rect['width'];
    position = xPos/box_rect['width'];

    // console.log("Pop-over width: "+popCoverage+", Position: "+position);

    //Find the start of the left edge depending on popOver's width and the position

    // Try to place on the right
    if (position + popCoverage + 0.1 < 1) {
        popOver.style.left = xPos + 'px';
    }
    //Try to place on the left
    else if (position > popCoverage + 0.1) {
        popOver.style.left = xPos - rect['width'] + 'px';
    }
    //Box doesn't fit properly in the left or right of the clicked position.
    else {
        offset = 1 - position - (popCoverage + 0.1);
        xPos = (xPos + offset * box_rect['width']) + 'px';
        popOver.style.left = xPos + 'px';
    }
    
    popOver.style.top = yPos + 'px';
    event.stopPropagation();
}

function parenthasClass(element, className) {
    if (element.tagName == 'BODY') {
        return false;
    }
    else if (element.classList.contains(className)) {
        return true;
    }
    return parenthasClass(element.parentElement, className);
}

window.addEventListener('DOMContentLoaded', function(event) {
    var triviaBtns = document.querySelectorAll('.trivia-button');
    var body = document.querySelector('body');

    triviaBtns.forEach(function(triviaBtn){
        triviaBtn.addEventListener('click', showPopOver);
    });
    
    body.addEventListener('click', function(e) {
        if (!parenthasClass(e.target, 'trivia-content')) {
            //Check the opened popOver
            if (openedPopOver != null) {
                hidePopOver(openedPopOver);
            } 
        }
    });

});
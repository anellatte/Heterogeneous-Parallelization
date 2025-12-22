#include <iostream>
#include <omp.h>
#include <chrono>

using namespace std;

/* ОДНОСВЯЗНЫЙ СПИСОК */

struct Node {
    int data;
    Node* next;
};

// Добавление элемента в начало списка
void addToList(Node*& head, int value) {
    Node* newNode = new Node;
    newNode->data = value;
    newNode->next = head;
    head = newNode;
}

// Поиск элемента в списке
bool searchInList(Node* head, int value) {
    Node* current = head;
    while (current != nullptr) {
        if (current->data == value)
            return true;
        current = current->next;
    }
    return false;
}

// Удаление элемента из списка
void deleteFromList(Node*& head, int value) {
    if (head == nullptr) return;

    if (head->data == value) {
        Node* temp = head;
        head = head->next;
        delete temp;
        return;
    }

    Node* current = head;
    while (current->next != nullptr && current->next->data != value) {
        current = current->next;
    }

    if (current->next != nullptr) {
        Node* temp = current->next;
        current->next = temp->next;
        delete temp;
    }
}

/* СТЕК (LIFO) */

struct Stack {
    int data[1000];
    int top = -1;
};

bool isEmptyStack(Stack& s) {
    return s.top == -1;
}

void push(Stack& s, int value) {
    s.data[++s.top] = value;
}

void pop(Stack& s) {
    if (!isEmptyStack(s))
        s.top--;
}

/*  ОЧЕРЕДЬ (FIFO) */

struct Queue {
    int data[100000];
    int front = 0;
    int rear = 0;
};

bool isEmptyQueue(Queue& q) {
    return q.front == q.rear;
}

void enqueue(Queue& q, int value) {
    q.data[q.rear++] = value;
}

void dequeue(Queue& q) {
    if (!isEmptyQueue(q))
        q.front++;
}

/* MAIN */

int main() {

    /* Односвязный список */
    Node* head = nullptr;

    addToList(head, 10);
    addToList(head, 20);
    addToList(head, 30);

    cout << "Поиск 20 в списке: "
         << (searchInList(head, 20) ? "Найдено" : "Не найдено") << endl;

    deleteFromList(head, 20);

    cout << "Поиск 20 после удаления: "
         << (searchInList(head, 20) ? "Найдено" : "Не найдено") << endl;


    /* Стек */
    Stack s;
    push(s, 1);
    push(s, 2);
    push(s, 3);

    pop(s);
    cout << "Стек пуст? "
         << (isEmptyStack(s) ? "Да" : "Нет") << endl;


    /* Очередь */
    Queue q;
    enqueue(q, 5);
    enqueue(q, 10);
    dequeue(q);

    cout << "Очередь пуста? "
         << (isEmptyQueue(q) ? "Да" : "Нет") << endl;


    /* ПАРАЛЛЕЛЬНОЕ ДОБАВЛЕНИЕ В ОЧЕРЕДЬ*/

    Queue pq;
    int N = 100000;

    auto start_seq = chrono::high_resolution_clock::now();

    for (int i = 0; i < N; i++) {
        enqueue(pq, i);
    }

    auto end_seq = chrono::high_resolution_clock::now();
    chrono::duration<double> seq_time = end_seq - start_seq;

    pq.front = pq.rear = 0;

    auto start_par = chrono::high_resolution_clock::now();

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        #pragma omp critical
        {
            enqueue(pq, i);
        }
    }

    auto end_par = chrono::high_resolution_clock::now();
    chrono::duration<double> par_time = end_par - start_par;

    cout << "\nПоследовательное добавление: "
         << seq_time.count() << " секунд" << endl;

    cout << "Параллельное добавление (OpenMP): "
         << par_time.count() << " секунд" << endl;

    return 0;
}
